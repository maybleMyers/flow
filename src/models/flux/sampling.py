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


def denoise_batched_timesteps(
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
    timesteps: Tensor,  # Shape: (B, N), where N is the number of time points
    guidance: float = 4.0,
):
    """
    Performs ODE solving using the Euler method with potentially different
    timestep sequences for each sample in the batch.

    Args:
        model: The flow matching model.
        img: Input tensor (e.g., noise) shape (B, C, H, W).
        img_ids: Image IDs tensor, shape (B, ...).
        txt: Text conditioning tensor, shape (B, L, D).
        txt_ids: Text IDs tensor, shape (B, L).
        txt_mask: Text mask tensor, shape (B, L).
        timesteps: Tensor containing the time points for each batch sample.
                   Shape (B, N), where B is the batch size and N is the
                   number of time points (e.g., [t_start, ..., t_end]).
                   Time should generally decrease (e.g., [1.0, 0.8, ..., 0.0]).
        guidance: Classifier-free guidance strength.
    Returns:
        Denoised image tensor, shape (B, C, H, W).
    """
    batch_size = img.shape[0]
    num_time_points = timesteps.shape[1]
    num_steps = num_time_points - 1  # Number of integration steps

    if timesteps.shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch: img has {batch_size}, "
            f"but timesteps has {timesteps.shape[0]}"
        )
    if timesteps.ndim != 2:
        raise ValueError(
            f"timesteps tensor must be 2D (B, N), but got shape {timesteps.shape}"
        )

    # Guidance vector remains the same for all elements in this specific call
    guidance_vec = torch.full(
        (batch_size,), guidance, device=img.device, dtype=img.dtype
    )

    # Ensure timesteps tensor is on the same device and dtype as img
    timesteps = timesteps.to(device=img.device, dtype=img.dtype)

    # Iterate through the integration steps (from step 0 to N-2)
    for i in range(num_steps):
        # Get the current time for each batch element
        t_curr_batch = timesteps[:, i]  # Shape: (B,)
        # Get the next time for each batch element
        t_next_batch = timesteps[:, i + 1]  # Shape: (B,)

        # Model prediction using the current time for each batch element
        # Your model already accepts batched timesteps (shape B,)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            txt_mask=txt_mask,
            timesteps=t_curr_batch,  # Pass the (B,) tensor of current times
            guidance=guidance_vec,
            y=clip_embedding,
        )

        # Calculate the step size (dt) for each batch element
        # dt = t_next - t_curr (Note: if time goes 1->0, dt will be negative)
        dt_batch = t_next_batch - t_curr_batch  # Shape: (B,)

        # Reshape dt for broadcasting: (B,) -> (B, 1, 1)
        dt_batch_reshaped = dt_batch.view(batch_size, 1, 1)

        # Euler step update: x_{t+1} = x_t + dt * v(x_t, t)
        img = img + dt_batch_reshaped * pred

    return img
