import argparse
import torch
import os
from torchvision.utils import save_image
from transformers import T5Tokenizer
from src.models.chroma.model_dct import Chroma, chroma_params
from src.models.chroma.module.t5 import T5EncoderModel, T5Config, replace_keys
from src.general_utils import load_file_multipart, load_safetensors
from src.trainer.train_chroma_dct import inference_wrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Run ChromaRadiance inference using inference_wrapper")
    
    # Core Paths
    parser.add_argument("--model_path", required=True, help="Path to the Chroma model (.pth or .safetensors)")
    parser.add_argument("--t5_path", required=True, help="Path to T5 model directory")
    parser.add_argument("--t5_config", required=True, help="Path to T5 config.json")
    parser.add_argument("--t5_tokenizer", required=True, help="Path to T5 tokenizer directory")

    # Generation Parameters
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", default="", help="Negative prompt. If empty, an unconditional embedding is used.")
    parser.add_argument("--output", default="output.png", help="Output image filename")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument("--cfg", type=float, default=5.0, help="Classifier-Free Guidance scale. This is the main control for prompt strength.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. If not set, a random seed will be used.")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--first_n_steps_no_cfg", type=int, default=-1, help="Number of initial steps to run without CFG. -1 means always use CFG (if cfg > 1.0).")

    # Performance & Device
    parser.add_argument("--device", default="cuda", help="Main device for the Chroma model (e.g., 'cuda', 'cpu')")
    parser.add_argument("--t5_device", default="cpu", help="Device for T5 encoder ('cuda' for faster, 'cpu' for lower VRAM). The wrapper expects T5 on CPU initially.")
    parser.add_argument("--t5_max_length", type=int, default=512, help="T5 maximum sequence length")

    return parser.parse_args()

def generate_intelligent_filename(base_filename, cfg, seed, width, height, steps):
    """
    Generate an intelligent filename that includes key generation parameters.
    """
    name, ext = os.path.splitext(base_filename)
    if not ext:
        ext = '.png'
    intelligent_name = f"{name}_cfg{cfg}_seed{seed}_{width}x{height}_steps{steps}{ext}"
    return intelligent_name

def main():
    args = parse_args()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    
    if args.seed is None:
        args.seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    # Set seed globally for reproducibility, although the wrapper uses its own generator.
    torch.manual_seed(args.seed)

    print("\n--- Inference Settings ---")
    print(f"Prompt: {args.prompt}")
    print(f"Negative Prompt: {args.negative_prompt if args.negative_prompt else '(Not provided)'}")
    print(f"Steps: {args.steps}, CFG Scale: {args.cfg}")
    print(f"Dimensions: {args.width}x{args.height}")
    print(f"Seed: {args.seed}")
    print("--------------------------\n")

    with torch.no_grad():
        # --- 2. Call the Inference Wrapper ---
        print(f"Running {args.steps} denoising steps via inference_wrapper...")

        image_dimensions_for_wrapper = (args.height, args.width)
        
        output_image = inference_wrapper(
            model=model,
            t5_tokenizer=tokenizer,
            t5=t5_model,
            seed=args.seed,
            steps=args.steps,
            guidance=0.0,  # CRITICAL: This is the internal guidance, MUST be 0.0 for this model.
            cfg=args.cfg,
            prompts=[args.prompt],  # The wrapper expects a list of prompts.
            negative_prompts=[args.negative_prompt],
            rank=args.device,      # The wrapper uses the 'rank' parameter as the target device.
            first_n_steps_wo_cfg=args.first_n_steps_no_cfg,
            image_dim=image_dimensions_for_wrapper,
            t5_max_length=args.t5_max_length,
        )

        # --- 3. Save Image ---
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