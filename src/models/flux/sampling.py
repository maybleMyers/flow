import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    latent_depth: int = 16,
    spatial_compression: int = 8,
):
    return torch.randn(
        num_samples,
        latent_depth,
        # allow for packing
        math.ceil(height / spatial_compression),
        math.ceil(width / spatial_compression),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    # guidance
    txt: Tensor,
    # guidance ID
    txt_ids: Tensor,
    # mask
    txt_mask: Tensor,
    clip_embedding: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 0.0,
):
    # this is ignored for schnell
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            txt_mask=txt_mask,
            timesteps=t_vec,
            guidance=guidance_vec,
            y=clip_embedding,
        )

        img = img + (t_prev - t_curr) * pred

    return img




def denoise_cfg(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    # guidance
    txt: Tensor,
    neg_txt: Tensor,
    # guidance ID
    txt_ids: Tensor,
    neg_txt_ids: Tensor,
    # mask
    txt_mask: Tensor,
    neg_txt_mask: Tensor,
    # clip embedding
    clip_embedding: Tensor,
    neg_clip_embedding: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    cfg: float = 2.0,
    first_n_steps_without_cfg: int = 4,
):
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )
    
    for step_count, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        
        apply_cfg = cfg > 1.0 and (step_count >= first_n_steps_without_cfg or first_n_steps_without_cfg == -1)

        if apply_cfg:
            # --- Positive Prediction ---
            pred_pos = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                txt_mask=txt_mask,
                timesteps=t_vec,
                guidance=guidance_vec,
                y=clip_embedding,
            )

            # --- Negative Prediction ---
            pred_neg = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                txt_mask=neg_txt_mask,
                timesteps=t_vec,
                guidance=guidance_vec,
                y=neg_clip_embedding,
            )
            
            # Combine using CFG formula: uncond + cfg * (cond - uncond)
            pred_final = pred_neg + cfg * (pred_pos - pred_neg)
        else:
            # --- Positive Prediction Only ---
            pred_final = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                txt_mask=txt_mask,
                timesteps=t_vec,
                guidance=guidance_vec,
                y=clip_embedding,
            )

        # Euler step update
        img = img.to(pred_final) + (t_prev - t_curr) * pred_final

    return img



def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )