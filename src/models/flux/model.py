from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.utils.checkpoint as ckpt

from .module.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    SingleStreamBlock,
    timestep_embedding,
    MLPEmbedder
)


@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    _use_compiled: bool


flux_params = FluxParams(
    in_channels=64,
    out_channels=64,
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    guidance_embed=True,
    _use_compiled=False
)


def modify_mask_to_attend_padding(mask, max_seq_length, num_extra_padding=8):
    """
    Modifies attention mask to allow attention to a few extra padding tokens.

    Args:
        mask: Original attention mask (1 for tokens to attend to, 0 for masked tokens)
        max_seq_length: Maximum sequence length of the model
        num_extra_padding: Number of padding tokens to unmask

    Returns:
        Modified mask
    """
    # Get the actual sequence length from the mask
    seq_length = mask.sum(dim=-1)
    batch_size = mask.shape[0]

    modified_mask = mask.clone()

    for i in range(batch_size):
        current_seq_len = int(seq_length[i].item())

        # Only add extra padding tokens if there's room
        if current_seq_len < max_seq_length:
            # Calculate how many padding tokens we can unmask
            available_padding = max_seq_length - current_seq_len
            tokens_to_unmask = min(num_extra_padding, available_padding)

            # Unmask the specified number of padding tokens right after the sequence
            modified_mask[i, current_seq_len : current_seq_len + tokens_to_unmask] = 1

    return modified_mask


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    use_compiled=params._use_compiled,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    use_compiled=params._use_compiled,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(
            self.hidden_size,
            1,
            self.out_channels,
            use_compiled=params._use_compiled,
        )

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        attn_padding: int = 0,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # compute mask
        # assume max seq length from the batched input

        max_len = txt.shape[1]

        # mask
        with torch.no_grad():
            txt_mask_w_padding = modify_mask_to_attend_padding(
                txt_mask, max_len, attn_padding
            )
            txt_img_mask = torch.cat(
                [
                    txt_mask_w_padding,
                    torch.ones([img.shape[0], img.shape[1]], device=txt_mask.device),
                ],
                dim=1,
            )
            txt_img_mask = txt_img_mask.float().T @ txt_img_mask.float()
            txt_img_mask = (
                txt_img_mask[None, None, ...]
                .repeat(txt.shape[0], self.num_heads, 1, 1)
                .int()
                .bool()
            )
            # txt_mask_w_padding[txt_mask_w_padding==False] = True

        for i, block in enumerate(self.double_blocks):
            # the guidance replaced by FFN output

            # just in case in different GPU for simple pipeline parallel
            if self.training:
                img, txt = ckpt.checkpoint(
                    block, img, txt, pe, vec, txt_img_mask
                )
            else:
                img, txt = block(
                    img=img, txt=txt, pe=pe, vec=vec, mask=txt_img_mask
                )

        img = torch.cat((txt, img), 1)
        for i, block in enumerate(self.single_blocks):
            if self.training:
                img = ckpt.checkpoint(block, img, pe, vec, txt_img_mask)
            else:
                img = block(img, pe=pe, vec=vec, mask=txt_img_mask)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
