#!/usr/bin/env python3
"""
ChromaRadiance Inference Script with Smart Offloading

Usage:
    python run_inference.py --model_path path/to/model.safetensors --prompt "your prompt here"
    
Example (Basic):
    python run_inference.py \
        --model_path "C:/forge/testing/models/Stable-diffusion/2025-08-23_06-10-39.safetensors" \
        --t5_path "models/flux/text_encoder_2" \
        --t5_config "models/flux/text_encoder_2/config.json" \
        --t5_tokenizer "models/flux/tokenizer_2" \
        --prompt "a beautiful sunset over the ocean" \
        --output "output.png" \
        --steps 20 \
        --guidance 3.0 \
        --cfg 2.0 \
        --seed 42

Example (Low VRAM):
    python run_inference.py \
        --model_path "model.safetensors" \
        --t5_path "text_encoder" --t5_config "config.json" --t5_tokenizer "tokenizer" \
        --prompt "a beautiful landscape" \
        --low_vram --offload_strategy aggressive --show_memory_stats

Example (Ultra Low VRAM - 6GB or less):
    python run_inference.py \
        --model_path "model.safetensors" \
        --t5_path "text_encoder" --t5_config "config.json" --t5_tokenizer "tokenizer" \
        --prompt "a beautiful landscape" \
        --ultra_low_vram --show_memory_stats
"""

import argparse
import torch
from torchvision.utils import save_image
from transformers import T5Tokenizer
from src.models.chroma.model_dct import Chroma, chroma_params
from src.models.chroma.sampling import get_schedule, denoise_cfg
from src.models.chroma.utils import prepare_latent_image_ids
from src.models.chroma.module.t5 import T5EncoderModel, T5Config, replace_keys
from src.models.chroma.offloading_wrapper import apply_smart_offloading
from src.general_utils import load_file_multipart, load_safetensors
import src.lora_and_quant as lora_and_quant

def parse_args():
    parser = argparse.ArgumentParser(description="Run ChromaRadiance inference")
    
    # Model paths
    parser.add_argument("--model_path", required=True, 
                       help="Path to the ChromaRadiance model (.safetensors or .pth)")
    parser.add_argument("--t5_path", required=True,
                       help="Path to T5 model directory")
    parser.add_argument("--t5_config", required=True,
                       help="Path to T5 config.json")
    parser.add_argument("--t5_tokenizer", required=True,
                       help="Path to T5 tokenizer directory")
    
    # Generation parameters
    parser.add_argument("--prompt", required=True,
                       help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", default="",
                       help="Negative prompt (default: empty)")
    parser.add_argument("--output", default="generated_image.png",
                       help="Output image path (default: generated_image.png)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Number of images to generate (default: 1)")
    
    # Model parameters
    parser.add_argument("--height", type=int, default=512,
                       help="Image height (default: 512)")
    parser.add_argument("--width", type=int, default=512,
                       help="Image width (default: 512)")
    parser.add_argument("--steps", type=int, default=20,
                       help="Number of denoising steps (default: 20)")
    parser.add_argument("--guidance", type=float, default=3.0,
                       help="Guidance strength (default: 3.0)")
    parser.add_argument("--cfg", type=float, default=2.0,
                       help="Classifier-free guidance scale (default: 2.0)")
    parser.add_argument("--first_n_steps_without_cfg", type=int, default=-1,
                       help="First N steps without CFG. -1 = always use CFG, 0 = use CFG from start (default: -1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--device", default="cuda",
                       help="Device to use (default: cuda)")
    parser.add_argument("--t5_max_length", type=int, default=512,
                       help="T5 maximum sequence length (default: 512)")
    parser.add_argument("--use_8bit_t5", action="store_true",
                       help="Use 8-bit quantization for T5 to save memory")
    parser.add_argument("--cpu_offload", action="store_true",
                       help="Offload T5 to CPU during generation to save VRAM")
    parser.add_argument("--low_vram", action="store_true",
                       help="Enable all low VRAM optimizations (8bit T5 + CPU offload)")
    parser.add_argument("--ultra_low_vram", action="store_true",
                       help="Enable ultra low VRAM mode (aggressive memory cleanup + smaller batch processing)")
    
    # Advanced offloading options
    parser.add_argument("--auto_offload", action="store_true",
                       help="Enable automatic smart offloading based on VRAM detection")
    parser.add_argument("--offload_strategy", type=str, default="auto",
                       choices=["auto", "conservative", "aggressive", "ultra"],
                       help="Offloading strategy: auto (default), conservative, aggressive, or ultra")
    parser.add_argument("--disable_offloading", action="store_true",
                       help="Disable all offloading (keep everything on GPU)")
    parser.add_argument("--memory_target_gb", type=float, default=None,
                       help="Target VRAM usage in GB (experimental)")
    parser.add_argument("--show_memory_stats", action="store_true",
                       help="Show detailed memory usage statistics")
    parser.add_argument("--force_cpu_t5", action="store_true",
                       help="Force T5 to stay on CPU throughout inference (saves VRAM)")
    
    
    return parser.parse_args()

def load_models(args):
    """Load ChromaRadiance and T5 models."""
    
    # Clear any existing CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Check available memory before starting
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"Available GPU memory before loading: {free_mem / (1024**3):.2f}GB / {total_mem / (1024**3):.2f}GB")
        
        if free_mem < 1024**3:  # Less than 1GB free
            print("WARNING: Very low GPU memory available. Consider:")
            print("  1. Restarting your Python session")
            print("  2. Freeing other GPU processes") 
            print("  3. Using --cpu_offload flag")
    
    # Apply low VRAM optimizations
    if args.ultra_low_vram:
        args.use_8bit_t5 = True
        args.cpu_offload = True
        args.force_cpu_t5 = True  # Force T5 to stay on CPU
        args.low_vram = True
        args.auto_offload = True
        if args.offload_strategy == "auto":
            args.offload_strategy = "ultra"
        print("Ultra low VRAM mode enabled: 8-bit T5 + CPU offloading + aggressive cleanup + ultra offloading")
    elif args.low_vram:
        args.use_8bit_t5 = True
        args.cpu_offload = True
        args.auto_offload = True
        if args.offload_strategy == "auto":
            args.offload_strategy = "aggressive"
        print("Low VRAM mode enabled: 8-bit T5 + CPU offloading + aggressive offloading")
    
    # Enable auto offload by default for reasonable VRAM savings
    if not args.disable_offloading and not hasattr(args, 'auto_offload'):
        args.auto_offload = True
    
    print("Loading ChromaRadiance model...")
    
    # Load ChromaRadiance model
    with torch.device("meta"):
        model = Chroma(chroma_params)
    
    # Load model weights - support both .safetensors and .pth formats
    if args.model_path.endswith('.safetensors'):
        state_dict = load_safetensors(args.model_path)
    elif args.model_path.endswith('.pth'):
        state_dict = torch.load(args.model_path, map_location='cpu')
    else:
        raise ValueError(f"Unsupported model format. Expected .safetensors or .pth, got: {args.model_path}")
    
    model.load_state_dict(state_dict, assign=True)
    
    # Clean up state dict to free memory
    del state_dict
    torch.cuda.empty_cache()
    
    model.to(args.device).eval()
    
    # Apply smart offloading if enabled
    if args.auto_offload and not args.disable_offloading:
        print(f"Applying smart offloading with strategy: {args.offload_strategy}")
        model = apply_smart_offloading(model, torch.device(args.device), args.offload_strategy)
        if args.show_memory_stats:
            model.memory_manager.print_memory_summary()
    
    print("Loading T5 text encoder...")
    
    # Load T5 tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.t5_tokenizer)
    
    # Load T5 model
    config = T5Config.from_json_file(args.t5_config)
    with torch.device("meta"):
        t5_model = T5EncoderModel(config)
    
    state_dict = replace_keys(load_file_multipart(args.t5_path))
    t5_model.load_state_dict(state_dict, assign=True)
    
    # Clean up T5 state dict to free memory
    del state_dict
    torch.cuda.empty_cache()
    
    # T5 loading strategy: Start on CPU for memory efficiency, move to GPU during encoding
    if args.cpu_offload and not args.force_cpu_t5:
        t5_storage_device = "cpu"  # Where T5 lives when not in use
        t5_work_device = args.device  # Where T5 goes for encoding
        print("T5 will use just-in-time GPU loading (CPU storage, GPU encoding)")
    elif args.force_cpu_t5:
        t5_storage_device = "cpu"
        t5_work_device = "cpu"
        print("T5 will stay on CPU permanently")
    else:
        t5_storage_device = args.device
        t5_work_device = args.device
        print(f"T5 will stay on GPU ({args.device})")
    
    # Load T5 to storage device initially
    t5_model.to(t5_storage_device).eval()
    
    # Apply quantization on storage device
    if args.use_8bit_t5:
        print(f"Quantizing T5 to 8-bit on {t5_storage_device}...")
        lora_and_quant.swap_linear_recursive(
            t5_model, lora_and_quant.Quantized8bitLinear, device=t5_storage_device
        )
    
    # Store the device strategy in the model for inference use
    t5_model._storage_device = t5_storage_device
    t5_model._work_device = t5_work_device
    
    return model, t5_model, tokenizer

def run_inference(model, t5_model, tokenizer, args):
    """Run inference with the loaded models."""
    
    print(f"Generating image for prompt: '{args.prompt}'")
    if args.negative_prompt:
        print(f"Using negative prompt: '{args.negative_prompt}'")
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if "cuda" in args.device else "cpu", 
                           dtype=torch.bfloat16):
            
            # Set random seed
            torch.manual_seed(args.seed)
            
            # Generate noise directly in pixel space
            noise = torch.randn(
                args.batch_size, 3, args.height, args.width,
                device=args.device, 
                dtype=torch.bfloat16,
                generator=torch.Generator(device=args.device).manual_seed(args.seed)
            )
            
            # Prepare image position IDs
            image_ids = prepare_latent_image_ids(args.batch_size, args.height, args.width, patch_size=16).to(args.device)
            
            # Get sampling schedule
            timesteps = get_schedule(args.steps, 3)  # Use channel dimension like training
            
            # Just-in-time T5 loading: Move T5 to work device if needed
            t5_storage_device = getattr(t5_model, '_storage_device', 'cpu')
            t5_work_device = getattr(t5_model, '_work_device', args.device)
            
            current_t5_device = next(t5_model.parameters()).device
            need_t5_movement = (t5_work_device != t5_storage_device)
            
            if need_t5_movement:
                print(f"Moving T5 from {current_t5_device} to {t5_work_device} for encoding...")
                import time
                move_start = time.perf_counter()
                t5_model.to(t5_work_device)
                torch.cuda.empty_cache()
                move_time = time.perf_counter() - move_start
                print(f"T5 movement took {move_time:.2f}s")
            
            t5_device = next(t5_model.parameters()).device
            print(f"T5 encoding on device: {t5_device}")
            
            text_input = tokenizer(
                [args.prompt] * args.batch_size,
                padding="max_length",
                max_length=args.t5_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move input tensors to T5 device
            text_input = {k: v.to(t5_device) for k, v in text_input.items()}
            
            # Run T5 encoding with proper dtype handling
            if t5_device.type == 'cpu':
                # CPU inference without autocast (handles 8-bit quantization properly)
                text_embed = t5_model(text_input['input_ids'], text_input['attention_mask'])
            else:
                # GPU inference with autocast for performance
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    text_embed = t5_model(text_input['input_ids'], text_input['attention_mask'])
            
            # Move embeddings to main device for model inference
            text_embed = text_embed.to(args.device)
            
            # Encode negative prompt (repeat for batch)
            neg_prompt = args.negative_prompt if args.negative_prompt else ""
            neg_input = tokenizer(
                [neg_prompt] * args.batch_size,
                padding="max_length", 
                max_length=args.t5_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move input tensors to T5 device
            neg_input = {k: v.to(t5_device) for k, v in neg_input.items()}
            
            # Run T5 encoding with proper dtype handling  
            if t5_device.type == 'cpu':
                # CPU inference without autocast (handles 8-bit quantization properly)
                neg_embed = t5_model(neg_input['input_ids'], neg_input['attention_mask'])
            else:
                # GPU inference with autocast for performance
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    neg_embed = t5_model(neg_input['input_ids'], neg_input['attention_mask'])
            
            # Move embeddings to main device for model inference
            neg_embed = neg_embed.to(args.device)
            
            # Store attention masks before cleaning up inputs
            pos_attention_mask = text_input['attention_mask'].to(args.device)
            neg_attention_mask = neg_input['attention_mask'].to(args.device)
            
            # Move T5 back to storage device to free GPU memory
            if need_t5_movement:
                print(f"Moving T5 back to {t5_storage_device} to free GPU memory...")
                move_start = time.perf_counter()
                t5_model.to(t5_storage_device)
                torch.cuda.empty_cache()
                move_time = time.perf_counter() - move_start
                print(f"T5 offload took {move_time:.2f}s - GPU memory freed for main model inference")
                
                # Show memory freed
                if args.show_memory_stats:
                    free_mem, total_mem = torch.cuda.mem_get_info()
                    print(f"GPU memory after T5 offload: {(total_mem - free_mem) / (1024**3):.2f}GB / {total_mem / (1024**3):.2f}GB")
            
            # Clean up T5 related variables
            del text_input, neg_input
            torch.cuda.empty_cache()  # Always safe to call
            
            # Text position IDs (dummy)
            text_ids = torch.zeros((args.batch_size, args.t5_max_length, 3), device=args.device)


            # Run denoising
            print(f"Running {args.steps} denoising steps...")
            
            output = denoise_cfg(
                model, noise, image_ids,
                text_embed, neg_embed, text_ids, text_ids,
                pos_attention_mask, neg_attention_mask,
                timesteps, guidance=args.guidance, cfg=args.cfg, 
                first_n_steps_without_cfg=args.first_n_steps_without_cfg
            )
            
            # Clean up memory after inference
            del text_embed, neg_embed, noise, image_ids, text_ids
            del pos_attention_mask, neg_attention_mask
            
            # Ultra aggressive cleanup for ultra low VRAM mode
            if args.ultra_low_vram:
                # Force garbage collection and empty cache multiple times
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            else:
                torch.cuda.empty_cache()
                
            # Show performance stats if smart offloading was used
            if hasattr(model, 'print_performance_summary'):
                if args.show_memory_stats:
                    model.print_performance_summary()
            
            # Convert from [-1,1] to [0,1] and save
            images = output.clamp(-1, 1).add(1).div(2)
            
            if args.batch_size == 1:
                save_image(images[0], args.output)
                print(f"Generated image saved as '{args.output}'")
            else:
                # Save individual images and a grid
                from torchvision.utils import make_grid
                import os
                
                base_name, ext = os.path.splitext(args.output)
                
                for i, img in enumerate(images):
                    filename = f"{base_name}_{i:03d}{ext}"
                    save_image(img, filename)
                    print(f"Generated image {i+1}/{args.batch_size} saved as '{filename}'")
                
                # Save grid
                grid = make_grid(images, nrow=int(args.batch_size**0.5) or 1, padding=2, normalize=False)
                grid_filename = f"{base_name}_grid{ext}"
                save_image(grid, grid_filename)
                print(f"Generated grid saved as '{grid_filename}'")

def main():
    args = parse_args()
    
    print("ChromaRadiance Inference")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"T5: {args.t5_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {args.output}")
    print(f"Size: {args.width}x{args.height}")
    print(f"Steps: {args.steps}")
    print(f"Guidance: {args.guidance}")
    print(f"CFG: {args.cfg}")
    print(f"Seed: {args.seed}")
    print("=" * 50)
    
    # Load models
    model, t5_model, tokenizer = load_models(args)
    
    # Run inference
    run_inference(model, t5_model, tokenizer, args)
    
    
    print("Inference complete!")
    
    # Show final memory summary if requested
    if args.show_memory_stats and hasattr(model, 'memory_manager'):
        print("\nFinal Memory Summary:")
        model.memory_manager.print_memory_summary()

if __name__ == "__main__":
    main()