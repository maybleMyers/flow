# START OF FILE: run_inference_fixed.py

#!/usr/bin/env python3
"""
Corrected ChromaRadiance Inference Script

This script fixes the critical issue where the internal 'guidance' parameter
must be set to 0.0, as the model was exclusively trained with that value.
Prompt adherence is controlled ONLY by the '--cfg' parameter.

Example Usage:
    python run_inference_fixed.py \
        --model_path "/path/to/your/model.pth" \
        --t5_path "/path/to/text_encoder_2" \
        --t5_config "/path/to/text_encoder_2/config.json" \
        --t5_tokenizer "/path/to/tokenizer_2" \
        --prompt "a beautiful sunset over the ocean, professional photograph" \
        --negative_prompt "blurry, low quality, cartoon" \
        --output "sunset.png" \
        --steps 30 \
        --cfg 4.5 \
        --seed 123 \
        --width 1024 --height 1024
"""

import argparse
import torch
import numpy as np
import scipy.stats
from torchvision.utils import save_image
from transformers import T5Tokenizer

# Import necessary components from your source files
from src.models.chroma.model_dct import Chroma, chroma_params
from src.models.chroma.sampling import get_schedule, denoise_cfg
from src.models.chroma.utils import prepare_latent_image_ids
from src.models.chroma.module.t5 import T5EncoderModel, T5Config, replace_keys
from src.general_utils import load_file_multipart, load_safetensors

def flux_time_shift(mu: float, sigma: float, t):
    """forge's flux_time_shift function used in Chroma models"""
    import math
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def create_flux_sigmas(shift=1.15, timesteps=10000):
    """Create Flux-style sigma schedule as used in forge for Chroma models"""
    # Generate timesteps from 1/timesteps to 1.0
    t_values = (torch.arange(1, timesteps + 1, 1) / timesteps)
    # Apply flux time shift to create sigma schedule
    sigmas = torch.tensor([flux_time_shift(shift, 1.0, t.item()) for t in t_values])
    return sigmas

def get_forge_beta_schedule(num_steps, alpha=0.6, beta=0.6, shift=1.15):
    """
    Exact implementation of forge's beta scheduler for Chroma models.
    Uses forge's sigma generation + beta distribution sampling.
    """
    print(f"Using forge Beta scheduler with alpha={alpha}, beta={beta}, shift={shift}")
    
    # Create forge's Flux sigma schedule (same as Chroma uses)
    sigmas = create_flux_sigmas(shift=shift)
    total_timesteps = len(sigmas) - 1
    
    # Create linear timesteps from 1 to 0 (excluding endpoint)
    ts = 1 - np.linspace(0, 1, num_steps, endpoint=False)
    
    # Use beta percent point function to transform timesteps
    ts_transformed = scipy.stats.beta.ppf(ts, alpha, beta)
    
    # Map to sigma indices and extract actual sigma values
    indices = np.rint(ts_transformed * total_timesteps).astype(int)
    indices = np.clip(indices, 0, total_timesteps - 1)
    
    # Extract sigma values and convert to timesteps
    selected_sigmas = sigmas[indices]
    timesteps = selected_sigmas.tolist()
    timesteps.append(0.0)  # Add final timestep
    
    return timesteps

def parse_args():
    parser = argparse.ArgumentParser(description="Run ChromaRadiance inference correctly")
    # Beta
    parser.add_argument("--scheduler", type=str, default="forge_beta", choices=["default", "forge_beta"],
                        help="The timestep scheduler to use. forge beta matches the working implementation.")
    parser.add_argument("--beta_alpha", type=float, default=0.6, help="Alpha parameter for the forge Beta scheduler.")
    parser.add_argument("--beta_beta", type=float, default=0.6, help="Beta parameter for the forge Beta scheduler.")
    parser.add_argument("--shift", type=float, default=1.15, help="Shift parameter for Flux/Chroma time schedule.")
    # Core Paths
    parser.add_argument("--model_path", required=True, help="Path to the Chroma model (.pth or .safetensors)")
    parser.add_argument("--t5_path", required=True, help="Path to T5 model directory")
    parser.add_argument("--t5_config", required=True, help="Path to T5 config.json")
    parser.add_argument("--t5_tokenizer", required=True, help="Path to T5 tokenizer directory")

    # Generation Parameters
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", default="", help="Negative prompt")
    parser.add_argument("--output", default="output.png", help="Output image filename")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument("--cfg", type=float, default=5.0, help="Classifier-Free Guidance scale. This is the main control for prompt strength.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. If not set, a random seed will be used.")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--t5_device", default="cpu", help="Device for T5 encoder ('cuda' for faster, 'cpu' for lower VRAM)")


    # Performance & Device
    parser.add_argument("--device", default="cuda", help="Main device for the Chroma model (e.g., 'cuda', 'cpu')")
    parser.add_argument("--t5_max_length", type=int, default=512, help="T5 maximum sequence length")

    return parser.parse_args()

def generate_intelligent_filename(base_filename, cfg, seed, width, height, steps):
    """
    Generate an intelligent filename that includes key generation parameters.
    
    Args:
        base_filename: The base filename (from --output parameter)
        cfg: CFG scale value
        seed: Random seed used
        width: Image width
        height: Image height
        steps: Number of denoising steps
    
    Returns:
        Enhanced filename with parameters
    """
    import os
    
    # Split the filename into name and extension
    name, ext = os.path.splitext(base_filename)
    
    # If no extension provided, default to .png
    if not ext:
        ext = '.png'
    
    # Create the intelligent filename
    intelligent_name = f"{name}_cfg{cfg}_seed{seed}_{width}x{height}_steps{steps}{ext}"
    
    return intelligent_name

def main():
    args = parse_args()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 1. Load Models ---
    print("Loading Chroma model...")
    with torch.device("meta"):
        model = Chroma(chroma_params)

    if args.model_path.endswith('.safetensors'):
        state_dict = load_safetensors(args.model_path)
    else:
        state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict, assign=True)
    model.to(args.device).eval().to(torch.bfloat16)
    del state_dict

    print(f"Loading T5 text encoder to '{args.t5_device}'...")
    tokenizer = T5Tokenizer.from_pretrained(args.t5_tokenizer)
    t5_config = T5Config.from_json_file(args.t5_config)
    with torch.device("meta"):
        t5_model = T5EncoderModel(t5_config)
    t5_state_dict = replace_keys(load_file_multipart(args.t5_path))
    t5_model.load_state_dict(t5_state_dict, assign=True)
    t5_model.to(args.t5_device).eval().to(torch.bfloat16)
    del t5_state_dict
    
    # Use a random seed if none is provided
    if args.seed is None:
        args.seed = torch.randint(0, 2**32 - 1, (1,)).item()
    torch.manual_seed(args.seed)

    print("\n--- Inference Settings ---")
    print(f"Prompt: {args.prompt}")
    print(f"Negative Prompt: {args.negative_prompt}")
    print(f"Steps: {args.steps}, CFG Scale: {args.cfg}")
    print(f"Dimensions: {args.width}x{args.height}")
    print(f"Seed: {args.seed}")
    print("--------------------------\n")

    with torch.no_grad():
        # --- 2. Encode Prompts with T5 ---
        print("Encoding text prompts...")
        # Positive prompt with forge tokenizer options
        text_inputs = tokenizer(
            [args.prompt], 
            padding="max_length", 
            max_length=max(args.t5_max_length, 3),  # min_length=3
            truncation=True, 
            return_tensors="pt",
            pad_to_multiple_of=None  # min_padding=0 equivalent
        ).to(args.t5_device)
        text_embed = t5_model(text_inputs.input_ids, text_inputs.attention_mask).to(args.device)
        pos_attention_mask = text_inputs.attention_mask.to(args.device)

        # Negative prompt with forge tokenizer options
        neg_inputs = tokenizer(
            [args.negative_prompt], 
            padding="max_length", 
            max_length=max(args.t5_max_length, 3),  # min_length=3
            truncation=True, 
            return_tensors="pt",
            pad_to_multiple_of=None  # min_padding=0 equivalent
        ).to(args.t5_device)
        neg_embed = t5_model(neg_inputs.input_ids, neg_inputs.attention_mask).to(args.device)
        neg_attention_mask = neg_inputs.attention_mask.to(args.device)
        
        # --- 3. Unload T5 to free VRAM ---
        print("Unloading T5 model to free VRAM...")
        del t5_model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- 4. Prepare for Denoising ---
        # Initial noise
        noise = torch.randn(
            1, 3, args.height, args.width, device=args.device, dtype=torch.bfloat16
        )

        # Positional IDs for the image
        image_ids = prepare_latent_image_ids(1, args.height, args.width, patch_size=16).to(args.device)
        
        # Dummy positional IDs for text (as in the original repo)
        text_ids = torch.zeros((1, args.t5_max_length, 3), device=args.device)
        neg_text_ids = torch.zeros((1, args.t5_max_length, 3), device=args.device)

        # Timestep schedule
        if args.scheduler == "forge_beta":
            timesteps = get_forge_beta_schedule(args.steps, alpha=args.beta_alpha, beta=args.beta_beta, shift=args.shift)
        else:
            # Fallback to the original scheduler from sampling.py
            timesteps = get_schedule(args.steps, 3) 
        # --- 5. Run Denoising Loop ---
        print(f"Running {args.steps} denoising steps...")
        output_image = denoise_cfg(
            model=model,
            img=noise,
            img_ids=image_ids,
            txt=text_embed,
            neg_txt=neg_embed,
            txt_ids=text_ids,
            neg_txt_ids=neg_text_ids,
            txt_mask=pos_attention_mask,
            neg_txt_mask=neg_attention_mask,
            timesteps=timesteps,
            guidance=0.0,  # CRITICAL: This MUST be 0.0 as the model was trained this way.
            cfg=args.cfg,
            first_n_steps_without_cfg=-1, # Always use CFG
        )

        # --- 6. Save Image ---
        # Generate intelligent filename with parameters
        intelligent_filename = generate_intelligent_filename(
            args.output, args.cfg, args.seed, args.width, args.height, args.steps
        )
        print(f"Saving image to {intelligent_filename}...")
        # Clamp, convert from [-1, 1] to [0, 1], and save
        output_image = output_image.clamp(-1, 1).add(1).div(2).to(torch.float32)
        save_image(output_image[0], intelligent_filename)
        
    print("\nInference complete!")


if __name__ == "__main__":
    main()