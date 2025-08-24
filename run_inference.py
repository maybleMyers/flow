#!/usr/bin/env python3
"""
ChromaRadiance Inference Script

Usage:
    python run_inference.py --model_path path/to/model.safetensors --prompt "your prompt here"
    
Example:
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
"""

import argparse
import torch
from torchvision.utils import save_image
from transformers import T5Tokenizer
from src.models.chroma.model_dct import Chroma, chroma_params
from src.models.chroma.sampling import get_schedule, denoise_cfg
from src.models.chroma.utils import prepare_latent_image_ids
from src.models.chroma.module.t5 import T5EncoderModel, T5Config, replace_keys
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
    
    
    return parser.parse_args()

def load_models(args):
    """Load ChromaRadiance and T5 models."""
    
    # Apply low VRAM optimizations
    if args.ultra_low_vram:
        args.use_8bit_t5 = True
        args.cpu_offload = True
        args.low_vram = True
        print("Ultra low VRAM mode enabled: 8-bit T5 + CPU offloading + aggressive cleanup")
    elif args.low_vram:
        args.use_8bit_t5 = True
        args.cpu_offload = True
        print("Low VRAM mode enabled: 8-bit T5 + CPU offloading")
    
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
    
    # Apply memory optimizations
    if args.use_8bit_t5:
        print("Quantizing T5 to 8-bit...")
        lora_and_quant.swap_linear_recursive(
            t5_model, lora_and_quant.Quantized8bitLinear, device=args.device
        )
    
    # Start T5 on CPU if offloading enabled
    if args.cpu_offload:
        print("T5 will be offloaded to CPU during generation")
        t5_model.to("cpu").eval()
    else:
        t5_model.to(args.device).eval()
    
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
            num_patches = (args.height // 16) * (args.width // 16)
            timesteps = get_schedule(args.steps, num_patches)
            
            # Handle T5 encoding with potential CPU offloading
            if args.cpu_offload:
                # Move T5 to GPU temporarily for encoding
                t5_model.to(args.device)
                torch.cuda.empty_cache()
            
            # Encode positive prompt (repeat for batch)
            t5_device = t5_model.device
            text_input = tokenizer(
                [args.prompt] * args.batch_size,
                padding="max_length",
                max_length=args.t5_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(t5_device)
            
            text_embed = t5_model(text_input.input_ids, text_input.attention_mask).to(args.device)
            
            # Encode negative prompt (repeat for batch)
            neg_prompt = args.negative_prompt if args.negative_prompt else ""
            neg_input = tokenizer(
                [neg_prompt] * args.batch_size,
                padding="max_length", 
                max_length=args.t5_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(t5_device)
            
            neg_embed = t5_model(neg_input.input_ids, neg_input.attention_mask).to(args.device)
            
            # Move T5 back to CPU if offloading enabled
            if args.cpu_offload:
                t5_model.to("cpu")
                # Clean up T5 related variables
                del text_input, neg_input
                torch.cuda.empty_cache()
            
            # Text position IDs (dummy)
            text_ids = torch.zeros((args.batch_size, args.t5_max_length, 3), device=args.device)


            # Run denoising
            print(f"Running {args.steps} denoising steps...")
            
            # Store attention masks before cleaning up inputs
            pos_attention_mask = text_embed.new_ones(args.batch_size, args.t5_max_length) if args.cpu_offload else text_input.attention_mask
            neg_attention_mask = text_embed.new_ones(args.batch_size, args.t5_max_length) if args.cpu_offload else neg_input.attention_mask
            
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
            if not args.cpu_offload:
                del text_input, neg_input
            
            # Ultra aggressive cleanup for ultra low VRAM mode
            if args.ultra_low_vram:
                # Force garbage collection and empty cache multiple times
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            else:
                torch.cuda.empty_cache()
            
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

if __name__ == "__main__":
    main()