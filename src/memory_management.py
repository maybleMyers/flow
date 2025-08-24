"""
Advanced Memory Management System for Chroma Inference
Inspired by ChromaForge's memory management with adaptations for run_inference.py
"""

import sys
import time
import psutil
import torch
import platform
from enum import Enum
from typing import List, Dict, Optional, Tuple
import gc


class VRAMState(Enum):
    DISABLED = 0    # No vram present: no need to move models to vram
    NO_VRAM = 1     # Very low vram: enable all the options to save vram
    LOW_VRAM = 2    # Low vram: offload some components
    NORMAL_VRAM = 3 # Normal vram: keep most on GPU
    HIGH_VRAM = 4   # High vram: keep everything on GPU


class MemoryManager:
    """Advanced memory manager for Chroma inference"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.cpu_device = torch.device('cpu')
        self.vram_state = self._detect_vram_state()
        self.loaded_components = {}  # Track what's currently on GPU
        
    def _detect_vram_state(self) -> VRAMState:
        """Detect VRAM state based on available memory"""
        if self.device.type == 'cpu':
            return VRAMState.DISABLED
            
        try:
            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)  # GB
                free_vram = torch.cuda.memory_stats(self.device).get('reserved_bytes.all.allocated', 0)
                free_vram = (torch.cuda.get_device_properties(self.device).total_memory - free_vram) / (1024**3)
                
                print(f"Total VRAM: {total_vram:.1f} GB, Free VRAM: {free_vram:.1f} GB")
                
                if total_vram >= 24:
                    return VRAMState.HIGH_VRAM
                elif total_vram >= 12:
                    return VRAMState.NORMAL_VRAM
                elif total_vram >= 8:
                    return VRAMState.LOW_VRAM
                else:
                    return VRAMState.NO_VRAM
            else:
                return VRAMState.DISABLED
                
        except Exception as e:
            print(f"Warning: Could not detect VRAM state: {e}")
            return VRAMState.LOW_VRAM  # Safe default
    
    def get_free_memory(self, device: Optional[torch.device] = None) -> int:
        """Get free memory in bytes"""
        if device is None:
            device = self.device
            
        if device.type == 'cpu':
            return psutil.virtual_memory().available
        elif device.type == 'cuda':
            try:
                free, total = torch.cuda.mem_get_info(device)
                return free
            except:
                return 0
        else:
            return 0
    
    def get_total_memory(self, device: Optional[torch.device] = None) -> int:
        """Get total memory in bytes"""
        if device is None:
            device = self.device
            
        if device.type == 'cpu':
            return psutil.virtual_memory().total
        elif device.type == 'cuda':
            try:
                free, total = torch.cuda.mem_get_info(device)
                return total
            except:
                return 0
        else:
            return 0
    
    def estimate_model_memory(self, model: torch.nn.Module) -> int:
        """Estimate memory usage of a model in bytes"""
        total_memory = 0
        for param in model.parameters():
            param_memory = param.numel() * param.element_size()
            total_memory += param_memory
        return total_memory
    
    def should_offload_component(self, component_name: str, component_memory: int) -> bool:
        """Determine if a component should be offloaded based on memory constraints"""
        free_memory = self.get_free_memory()
        
        # Always offload in NO_VRAM mode
        if self.vram_state == VRAMState.NO_VRAM:
            return True
            
        # In LOW_VRAM mode, offload if component uses more than 25% of free memory
        elif self.vram_state == VRAMState.LOW_VRAM:
            return component_memory > (free_memory * 0.25)
            
        # In NORMAL_VRAM mode, only offload very large components
        elif self.vram_state == VRAMState.NORMAL_VRAM:
            return component_memory > (free_memory * 0.5)
            
        # In HIGH_VRAM mode, keep everything on GPU
        else:
            return False
    
    def move_to_device(self, model: torch.nn.Module, device: torch.device, 
                      component_name: str = "model") -> torch.nn.Module:
        """Move model to device with memory tracking"""
        try:
            current_device = next(model.parameters()).device
        except StopIteration:
            # No parameters, assume CPU
            current_device = torch.device('cpu')
            
        if current_device == device:
            return model
            
        print(f"Moving {component_name} from {current_device} to {device}")
        
        # Move model
        model = model.to(device)
        
        # Track loaded components
        if device == self.device:  # GPU
            self.loaded_components[component_name] = model
        elif component_name in self.loaded_components:
            del self.loaded_components[component_name]
            
        # Clean up memory
        if device == self.cpu_device:
            torch.cuda.empty_cache()
            
        return model
    
    def soft_empty_cache(self):
        """Empty CUDA cache safely"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    def aggressive_cleanup(self):
        """Perform aggressive memory cleanup"""
        # Force garbage collection
        gc.collect()
        
        # Empty CUDA cache multiple times
        for _ in range(3):
            self.soft_empty_cache()
            torch.cuda.synchronize()
    
    def get_memory_summary(self) -> Dict[str, float]:
        """Get current memory usage summary"""
        summary = {}
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_free, gpu_total = torch.cuda.mem_get_info()
            summary['gpu_free_gb'] = gpu_free / (1024**3)
            summary['gpu_total_gb'] = gpu_total / (1024**3)
            summary['gpu_used_gb'] = (gpu_total - gpu_free) / (1024**3)
            summary['gpu_usage_pct'] = ((gpu_total - gpu_free) / gpu_total) * 100
            
        # CPU memory  
        cpu_memory = psutil.virtual_memory()
        summary['cpu_free_gb'] = cpu_memory.available / (1024**3)
        summary['cpu_total_gb'] = cpu_memory.total / (1024**3)
        summary['cpu_used_gb'] = (cpu_memory.total - cpu_memory.available) / (1024**3)
        summary['cpu_usage_pct'] = cpu_memory.percent
        
        return summary
    
    def print_memory_summary(self):
        """Print current memory usage"""
        summary = self.get_memory_summary()
        print(f"Memory Status - VRAM State: {self.vram_state.name}")
        
        if 'gpu_total_gb' in summary:
            print(f"GPU: {summary['gpu_used_gb']:.2f}/{summary['gpu_total_gb']:.2f} GB "
                  f"({summary['gpu_usage_pct']:.1f}%)")
        
        print(f"CPU: {summary['cpu_used_gb']:.2f}/{summary['cpu_total_gb']:.2f} GB "
              f"({summary['cpu_usage_pct']:.1f}%)")
        
        if self.loaded_components:
            print(f"GPU Components: {list(self.loaded_components.keys())}")


class ChromaComponentManager:
    """Manages offloading of individual Chroma model components"""
    
    def __init__(self, model, memory_manager: MemoryManager):
        self.model = model
        self.memory_manager = memory_manager
        self.component_locations = {}  # Track where each component is located
        self.component_sizes = {}     # Cache component memory sizes
        
        # Analyze model structure
        self._analyze_components()
    
    def _analyze_components(self):
        """Analyze model components and their memory usage"""
        print("Analyzing Chroma model components...")
        
        # Helper function to get device safely
        def get_component_device(component):
            try:
                return next(component.parameters()).device
            except StopIteration:
                # No parameters, assume it's on the same device as the model
                return next(self.model.parameters()).device
        
        # Analyze double blocks
        for i, block in enumerate(self.model.double_blocks):
            name = f"double_block_{i}"
            size = self.memory_manager.estimate_model_memory(block)
            self.component_sizes[name] = size
            self.component_locations[name] = get_component_device(block)
            
        # Analyze single blocks  
        for i, block in enumerate(self.model.single_blocks):
            name = f"single_block_{i}"
            size = self.memory_manager.estimate_model_memory(block)
            self.component_sizes[name] = size
            self.component_locations[name] = get_component_device(block)
            
        # Analyze nerf blocks
        for i, block in enumerate(self.model.nerf_blocks):
            name = f"nerf_block_{i}"
            size = self.memory_manager.estimate_model_memory(block)
            self.component_sizes[name] = size
            self.component_locations[name] = get_component_device(block)
            
        # Analyze other components
        components = [
            ('distilled_guidance', self.model.distilled_guidance_layer),
            ('img_in_patch', self.model.img_in_patch),
            ('txt_in', self.model.txt_in),
            ('nerf_final_layer', self.model.nerf_final_layer_conv),
            ('pe_embedder', self.model.pe_embedder),
            ('nerf_embedder', self.model.nerf_image_embedder)
        ]
        
        for name, component in components:
            size = self.memory_manager.estimate_model_memory(component)
            self.component_sizes[name] = size
            self.component_locations[name] = get_component_device(component)
        
        # Print analysis
        total_size_mb = sum(self.component_sizes.values()) / (1024**2)
        print(f"Total model size: {total_size_mb:.1f} MB")
        
        # Show largest components
        sorted_components = sorted(self.component_sizes.items(), 
                                 key=lambda x: x[1], reverse=True)[:10]
        print("Largest components:")
        for name, size in sorted_components:
            print(f"  {name}: {size / (1024**2):.1f} MB")
    
    def apply_offloading_strategy(self, strategy: str = "auto"):
        """Apply offloading strategy based on VRAM state"""
        print(f"Applying offloading strategy: {strategy}")
        
        if strategy == "auto":
            self._auto_offload_strategy()
        elif strategy == "conservative":
            self._conservative_offload_strategy()
        elif strategy == "aggressive":
            self._aggressive_offload_strategy()
        elif strategy == "ultra":
            self._ultra_offload_strategy()
    
    def _auto_offload_strategy(self):
        """Automatically determine what to offload based on VRAM state"""
        vram_state = self.memory_manager.vram_state
        
        if vram_state == VRAMState.HIGH_VRAM:
            # Keep everything on GPU
            return
            
        elif vram_state == VRAMState.NORMAL_VRAM:
            # Offload largest single and double blocks
            self._offload_largest_blocks(num_blocks=5)
            
        elif vram_state == VRAMState.LOW_VRAM:
            # Offload most transformer blocks
            self._offload_largest_blocks(num_blocks=15)
            
        elif vram_state == VRAMState.NO_VRAM:
            # Offload everything possible
            self._ultra_offload_strategy()
    
    def _conservative_offload_strategy(self):
        """Conservative offloading - only offload if necessary"""
        # Only offload the largest components that exceed memory threshold
        for name, size in sorted(self.component_sizes.items(), 
                               key=lambda x: x[1], reverse=True):
            if self.memory_manager.should_offload_component(name, size):
                self._offload_component(name)
    
    def _aggressive_offload_strategy(self):
        """Aggressive offloading - offload many transformer blocks"""
        # Offload most double and single blocks
        self._offload_component_type("double_block")
        
        # Keep only a few single blocks on GPU
        single_blocks_to_keep = 10
        single_block_names = [name for name in self.component_sizes.keys() 
                             if name.startswith("single_block")]
        sorted_single = sorted(single_block_names, 
                             key=lambda x: self.component_sizes[x])
        
        for name in sorted_single[single_blocks_to_keep:]:
            self._offload_component(name)
    
    def _ultra_offload_strategy(self):
        """Ultra aggressive offloading - offload almost everything"""
        # Offload all transformer blocks
        self._offload_component_type("double_block")
        self._offload_component_type("single_block")
        self._offload_component_type("nerf_block")
        
        # Offload distilled guidance layer
        self._offload_component("distilled_guidance")
    
    def _offload_largest_blocks(self, num_blocks: int):
        """Offload the N largest transformer blocks"""
        transformer_blocks = [name for name in self.component_sizes.keys() 
                            if any(x in name for x in ["double_block", "single_block"])]
        
        sorted_blocks = sorted(transformer_blocks, 
                              key=lambda x: self.component_sizes[x], reverse=True)
        
        for name in sorted_blocks[:num_blocks]:
            self._offload_component(name)
    
    def _offload_component_type(self, component_type: str):
        """Offload all components of a specific type"""
        matching_components = [name for name in self.component_sizes.keys() 
                              if component_type in name]
        
        for name in matching_components:
            self._offload_component(name)
    
    def _offload_component(self, component_name: str):
        """Offload a specific component to CPU"""
        if component_name.startswith("double_block"):
            idx = int(component_name.split("_")[2])
            block = self.model.double_blocks[idx]
            self.model.double_blocks[idx] = self.memory_manager.move_to_device(
                block, self.memory_manager.cpu_device, component_name)
                
        elif component_name.startswith("single_block"):
            idx = int(component_name.split("_")[2])
            block = self.model.single_blocks[idx]
            self.model.single_blocks[idx] = self.memory_manager.move_to_device(
                block, self.memory_manager.cpu_device, component_name)
                
        elif component_name.startswith("nerf_block"):
            idx = int(component_name.split("_")[2])
            block = self.model.nerf_blocks[idx]
            self.model.nerf_blocks[idx] = self.memory_manager.move_to_device(
                block, self.memory_manager.cpu_device, component_name)
                
        elif component_name == "distilled_guidance":
            self.model.distilled_guidance_layer = self.memory_manager.move_to_device(
                self.model.distilled_guidance_layer, self.memory_manager.cpu_device, component_name)
        
        # Update tracking
        self.component_locations[component_name] = self.memory_manager.cpu_device
    
    def ensure_component_on_gpu(self, component_name: str):
        """Ensure a specific component is on GPU for computation"""
        if self.component_locations.get(component_name) == self.memory_manager.cpu_device:
            
            if component_name.startswith("double_block"):
                idx = int(component_name.split("_")[2])
                self.model.double_blocks[idx] = self.memory_manager.move_to_device(
                    self.model.double_blocks[idx], self.memory_manager.device, component_name)
                    
            elif component_name.startswith("single_block"):
                idx = int(component_name.split("_")[2])
                self.model.single_blocks[idx] = self.memory_manager.move_to_device(
                    self.model.single_blocks[idx], self.memory_manager.device, component_name)
                    
            elif component_name.startswith("nerf_block"):
                idx = int(component_name.split("_")[2])
                self.model.nerf_blocks[idx] = self.memory_manager.move_to_device(
                    self.model.nerf_blocks[idx], self.memory_manager.device, component_name)
                    
            elif component_name == "distilled_guidance":
                self.model.distilled_guidance_layer = self.memory_manager.move_to_device(
                    self.model.distilled_guidance_layer, self.memory_manager.device, component_name)
            
            # Update tracking
            self.component_locations[component_name] = self.memory_manager.device
    
    def get_offloading_summary(self) -> Dict[str, int]:
        """Get summary of what's offloaded"""
        gpu_count = sum(1 for loc in self.component_locations.values() 
                       if loc.type != 'cpu')
        cpu_count = sum(1 for loc in self.component_locations.values() 
                       if loc.type == 'cpu')
        
        return {
            'total_components': len(self.component_locations),
            'gpu_components': gpu_count,
            'cpu_components': cpu_count,
            'offload_percentage': (cpu_count / len(self.component_locations)) * 100
        }