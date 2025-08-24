"""
Smart Offloading Wrapper for Chroma Model
Provides just-in-time loading of model components during inference
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional
import time
from contextlib import contextmanager

from .model_dct import Chroma
from ...memory_management import MemoryManager, ChromaComponentManager


class ChromaOffloadingWrapper:
    """
    Wrapper around Chroma model that provides smart component offloading
    and just-in-time loading during inference
    """
    
    def __init__(self, model: Chroma, device: torch.device, offloading_strategy: str = "auto"):
        self.model = model
        self.device = device
        self.offloading_strategy = offloading_strategy
        
        # Initialize memory management
        self.memory_manager = MemoryManager(device)
        self.component_manager = ChromaComponentManager(model, self.memory_manager)
        
        # Apply initial offloading strategy
        self.component_manager.apply_offloading_strategy(offloading_strategy)
        
        # Track timing for performance analysis
        self.timing_stats = {
            'component_moves': 0,
            'move_time': 0.0,
            'inference_time': 0.0
        }
        
        print(f"ChromaOffloadingWrapper initialized with strategy: {offloading_strategy}")
        summary = self.component_manager.get_offloading_summary()
        print(f"Offloading summary: {summary['cpu_components']}/{summary['total_components']} "
              f"components on CPU ({summary['offload_percentage']:.1f}%)")
    
    @contextmanager
    def ensure_components_on_gpu(self, component_names: List[str]):
        """Context manager to ensure specific components are on GPU during computation"""
        moved_components = []
        start_time = time.perf_counter()
        
        # Move required components to GPU
        for component_name in component_names:
            if self.component_manager.component_locations.get(component_name, self.device) != self.device:
                self.component_manager.ensure_component_on_gpu(component_name)
                moved_components.append(component_name)
                self.timing_stats['component_moves'] += 1
        
        move_time = time.perf_counter() - start_time
        self.timing_stats['move_time'] += move_time
        
        if moved_components:
            print(f"Moved {len(moved_components)} components to GPU in {move_time:.3f}s")
        
        try:
            yield
        finally:
            # Optionally move components back to CPU for aggressive memory saving
            if self.offloading_strategy in ["ultra", "aggressive"]:
                for component_name in moved_components:
                    self._offload_component_after_use(component_name)
    
    def _offload_component_after_use(self, component_name: str):
        """Move component back to CPU after use in ultra-aggressive mode"""
        if component_name.startswith("double_block"):
            idx = int(component_name.split("_")[2])
            self.model.double_blocks[idx] = self.memory_manager.move_to_device(
                self.model.double_blocks[idx], self.memory_manager.cpu_device, component_name)
        elif component_name.startswith("single_block"):
            idx = int(component_name.split("_")[2])
            self.model.single_blocks[idx] = self.memory_manager.move_to_device(
                self.model.single_blocks[idx], self.memory_manager.cpu_device, component_name)
        elif component_name.startswith("nerf_block"):
            idx = int(component_name.split("_")[2])
            self.model.nerf_blocks[idx] = self.memory_manager.move_to_device(
                self.model.nerf_blocks[idx], self.memory_manager.cpu_device, component_name)
        
        self.component_manager.component_locations[component_name] = self.memory_manager.cpu_device
    
    def forward(self, img: Tensor, img_ids: Tensor, txt: Tensor, txt_ids: Tensor, 
                txt_mask: Tensor, timesteps: Tensor, guidance: Tensor, 
                attn_padding: int = 1) -> Tensor:
        """Forward pass with smart component loading"""
        
        start_time = time.perf_counter()
        
        # Ensure core components are on GPU
        core_components = ['distilled_guidance', 'img_in_patch', 'txt_in', 'pe_embedder', 'nerf_embedder']
        with self.ensure_components_on_gpu(core_components):
            
            # Initial processing
            if img.ndim != 4:
                raise ValueError("Input img tensor must be in [B, C, H, W] format.")
            if txt.ndim != 3:
                raise ValueError("Input txt tensors must have 3 dimensions.")
            B, C, H, W = img.shape

            # Store raw pixel values for NeRF head
            nerf_pixels = torch.nn.functional.unfold(img, 
                kernel_size=self.model.params.patch_size, 
                stride=self.model.params.patch_size)
            nerf_pixels = nerf_pixels.transpose(1, 2)
            
            # Patchify operations
            img = self.model.img_in_patch(img)
            num_patches = img.shape[2] * img.shape[3]
            img = img.flatten(2).transpose(1, 2)

            txt = self.model.txt_in(txt)

            # Generate modulation vectors (ensure distilled guidance is on GPU)
            with torch.no_grad():
                from .module.layers import timestep_embedding, distribute_modulations
                distill_timestep = timestep_embedding(timesteps, self.model.approximator_in_dim//4)
                distil_guidance = timestep_embedding(guidance, self.model.approximator_in_dim//4)
                modulation_index = timestep_embedding(self.model.mod_index, self.model.approximator_in_dim//2)
                modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
                timestep_guidance = (
                    torch.cat([distill_timestep, distil_guidance], dim=1)
                    .unsqueeze(1)
                    .repeat(1, self.model.mod_index_length, 1)
                )
                input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1)
                mod_vectors = self.model.distilled_guidance_layer(input_vec.requires_grad_(True))
            
            # Distribute modulation vectors  
            mod_vectors_dict = distribute_modulations(
                mod_vectors, self.model.depth_single_blocks, self.model.depth_double_blocks)

            # Position embeddings
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = self.model.pe_embedder(ids)

            # Compute attention mask
            max_len = txt.shape[1]
            with torch.no_grad():
                from .model_dct import modify_mask_to_attend_padding
                txt_mask_w_padding = modify_mask_to_attend_padding(
                    txt_mask, max_len, attn_padding)
                txt_img_mask = torch.cat([
                    txt_mask_w_padding,
                    torch.ones([img.shape[0], img.shape[1]], device=txt_mask.device),
                ], dim=1)
                txt_img_mask = txt_img_mask.float().T @ txt_img_mask.float()
                txt_img_mask = (
                    txt_img_mask[None, None, ...]
                    .repeat(txt.shape[0], self.model.num_heads, 1, 1)
                    .int().bool()
                )

        # Process double blocks with smart loading
        for i, block in enumerate(self.model.double_blocks):
            component_name = f"double_block_{i}"
            with self.ensure_components_on_gpu([component_name]):
                
                img_mod = mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"]
                txt_mod = mod_vectors_dict[f"double_blocks.{i}.txt_mod.lin"]
                double_mod = [img_mod, txt_mod]

                if self.model.training:
                    img, txt = torch.utils.checkpoint.checkpoint(
                        block, img, txt, pe, double_mod, txt_img_mask)
                else:
                    img, txt = block(
                        img=img, txt=txt, pe=pe, distill_vec=double_mod, mask=txt_img_mask)

        # Combine text and image sequences
        img = torch.cat((txt, img), 1)
        
        # Process single blocks with smart loading
        for i, block in enumerate(self.model.single_blocks):
            component_name = f"single_block_{i}"
            with self.ensure_components_on_gpu([component_name]):
                
                single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]
                if self.model.training:
                    img = torch.utils.checkpoint.checkpoint(block, img, pe, single_mod, txt_img_mask)
                else:
                    img = block(img, pe=pe, distill_vec=single_mod, mask=txt_img_mask)
        
        # Extract image tokens
        img = img[:, txt.shape[1]:, ...]

        # NeRF processing with smart loading
        nerf_components = ['nerf_embedder'] + [f"nerf_block_{i}" for i in range(len(self.model.nerf_blocks))] + ['nerf_final_layer']
        with self.ensure_components_on_gpu(nerf_components):
            
            # Prepare for per-patch processing
            nerf_hidden = img
            nerf_hidden = nerf_hidden.reshape(B * num_patches, self.model.params.hidden_size)
            nerf_pixels = nerf_pixels.reshape(B * num_patches, C, self.model.params.patch_size**2).transpose(1, 2)

            # Get DCT-encoded pixel embeddings
            img_dct = self.model.nerf_image_embedder(nerf_pixels)

            # Process through NeRF blocks
            for i, block in enumerate(self.model.nerf_blocks):
                if self.model.training:
                    img_dct = torch.utils.checkpoint.checkpoint(block, img_dct, nerf_hidden)
                else:
                    img_dct = block(img_dct, nerf_hidden)

            # Final projection
            img_dct = self.model.nerf_final_layer_conv.norm(img_dct)
            
            # Reassemble patches
            img_dct = img_dct.transpose(1, 2)
            img_dct = img_dct.reshape(B, num_patches, -1)
            img_dct = img_dct.transpose(1, 2)
            img_dct = torch.nn.functional.fold(
                img_dct,
                output_size=(H, W),
                kernel_size=self.model.params.patch_size,
                stride=self.model.params.patch_size
            )
            img_dct = self.model.nerf_final_layer_conv.conv(img_dct)

        inference_time = time.perf_counter() - start_time
        self.timing_stats['inference_time'] += inference_time
        
        return img_dct
    
    def __call__(self, *args, **kwargs):
        """Make wrapper callable like the original model"""
        return self.forward(*args, **kwargs)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        total_time = self.timing_stats['inference_time']
        move_time = self.timing_stats['move_time']
        compute_time = total_time - move_time
        
        return {
            'total_inference_time': total_time,
            'compute_time': compute_time,
            'move_time': move_time,
            'move_overhead_pct': (move_time / total_time * 100) if total_time > 0 else 0,
            'component_moves': self.timing_stats['component_moves']
        }
    
    def print_performance_summary(self):
        """Print performance summary"""
        stats = self.get_performance_stats()
        offload_summary = self.component_manager.get_offloading_summary()
        
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Offloading Strategy: {self.offloading_strategy}")
        print(f"Components on CPU: {offload_summary['cpu_components']}/{offload_summary['total_components']} "
              f"({offload_summary['offload_percentage']:.1f}%)")
        print(f"Total Inference Time: {stats['total_inference_time']:.3f}s")
        print(f"Compute Time: {stats['compute_time']:.3f}s")
        print(f"Model Movement Time: {stats['move_time']:.3f}s")
        print(f"Movement Overhead: {stats['move_overhead_pct']:.1f}%")
        print(f"Component Moves: {stats['component_moves']}")
        
        # Memory summary
        print("\nMEMORY USAGE:")
        self.memory_manager.print_memory_summary()
        print("="*50)
    
    def optimize_for_next_inference(self):
        """Optimize component placement for next inference based on usage patterns"""
        # This could be enhanced with usage tracking to intelligently pre-load
        # frequently used components
        pass
    
    def change_offloading_strategy(self, new_strategy: str):
        """Change the offloading strategy dynamically"""
        print(f"Changing offloading strategy from {self.offloading_strategy} to {new_strategy}")
        self.offloading_strategy = new_strategy
        self.component_manager.apply_offloading_strategy(new_strategy)
        
        summary = self.component_manager.get_offloading_summary()
        print(f"New offloading: {summary['cpu_components']}/{summary['total_components']} "
              f"components on CPU ({summary['offload_percentage']:.1f}%)")


def apply_smart_offloading(model: Chroma, device: torch.device, 
                          strategy: str = "auto") -> ChromaOffloadingWrapper:
    """
    Apply smart offloading to a Chroma model
    
    Args:
        model: The Chroma model to wrap
        device: The target device (usually GPU)
        strategy: Offloading strategy ("auto", "conservative", "aggressive", "ultra")
        
    Returns:
        ChromaOffloadingWrapper instance
    """
    return ChromaOffloadingWrapper(model, device, strategy)