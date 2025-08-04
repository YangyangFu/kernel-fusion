#!/usr/bin/env python3
"""
Stream-Aware Automatic Model Conversion

Extension of the automatic conversion system to support multi-stream execution.
This allows converted models to take advantage of CUDA streams for parallel
execution of independent operations.

Features:
- Stream-aware fused modules
- Parallel execution of independent layers
- Automatic stream allocation for model parts
- Stream synchronization management
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
import copy

from kernel_fusion.auto_convert import (
    ModelConverter, FusedLinearReLU, FusedLinearGELU, 
    FusedConv2dBNReLU, FusedLayerNormReLU, PatternDetector
)
from kernel_fusion.streams import StreamContext, StreamPriority


class StreamAwareFusedLinearReLU(FusedLinearReLU):
    """Stream-aware fused Linear + ReLU module"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preferred_stream = None
    
    def forward(self, x: torch.Tensor, stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        """Forward pass with optional stream specification"""
        target_stream = stream or self.preferred_stream
        
        if target_stream is not None and torch.cuda.is_available():
            with torch.cuda.stream(target_stream):
                return super().forward(x)
        else:
            return super().forward(x)


class StreamAwareFusedConv2dBNReLU(FusedConv2dBNReLU):
    """Stream-aware fused Conv2d + BatchNorm + ReLU module"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preferred_stream = None
    
    def forward(self, x: torch.Tensor, stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        """Forward pass with optional stream specification"""
        target_stream = stream or self.preferred_stream
        
        if target_stream is not None and torch.cuda.is_available():
            with torch.cuda.stream(target_stream):
                return super().forward(x)
        else:
            return super().forward(x)


class ParallelBlock(nn.Module):
    """Execute multiple modules in parallel using different streams"""
    
    def __init__(self, modules: List[nn.Module], stream_context: StreamContext):
        super().__init__()
        self.modules = nn.ModuleList(modules)
        self.stream_context = stream_context
        self.streams = []
        
        # Create a stream for each module
        for i, module in enumerate(modules):
            stream = stream_context.create_stream(f"parallel_block_{i}")
            self.streams.append(stream)
            
            # Assign preferred stream to stream-aware modules
            if hasattr(module, 'preferred_stream'):
                module.preferred_stream = stream
    
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Execute modules in parallel"""
        if len(inputs) != len(self.modules):
            raise ValueError(f"Number of inputs ({len(inputs)}) must match number of modules ({len(self.modules)})")
        
        results = []
        events = []
        
        # Launch all modules in parallel
        for i, (module, input_tensor, stream) in enumerate(zip(self.modules, inputs, self.streams)):
            event = self.stream_context.create_event(f"parallel_block_complete_{i}")
            
            with self.stream_context.stream(stream):
                if hasattr(module, 'forward') and 'stream' in module.forward.__code__.co_varnames:
                    result = module(input_tensor, stream=stream)
                else:
                    result = module(input_tensor)
                results.append(result)
                self.stream_context.record_event(f"parallel_block_complete_{i}", stream)
            
            events.append(f"parallel_block_complete_{i}")
        
        # Wait for all to complete
        for event in events:
            self.stream_context.wait_event(event)
        
        return results


class StreamAwareModelConverter(ModelConverter):
    """Extended model converter with stream awareness"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_context = None
        self.enable_parallel_execution = False
    
    def convert_model_with_streams(self, model: nn.Module,
                                  input_shape: Optional[Tuple[int, ...]] = None,
                                  device: str = 'cuda',
                                  enable_parallel: bool = True) -> Tuple[nn.Module, StreamContext]:
        """
        Convert model with stream support
        
        Returns:
            Tuple of (converted_model, stream_context)
        """
        self.enable_parallel_execution = enable_parallel
        self.stream_context = StreamContext(device=0 if device == 'cuda' else None)
        
        # Perform standard conversion first
        converted_model = self.convert_model(model, input_shape, device)
        
        if enable_parallel and torch.cuda.is_available():
            # Analyze model for parallelization opportunities
            converted_model = self._add_parallel_execution(converted_model)
        
        return converted_model, self.stream_context
    
    def _create_stream_aware_module(self, original_module, module_type: str):
        """Create stream-aware version of fused module"""
        if module_type == "linear_relu":
            return StreamAwareFusedLinearReLU(
                original_module.in_features,
                original_module.out_features,
                bias=original_module.bias is not None,
                device=original_module.weight.device,
                dtype=original_module.weight.dtype
            )
        elif module_type == "conv_bn_relu":
            return StreamAwareFusedConv2dBNReLU(
                original_module[0],  # conv
                original_module[1]   # bn
            )
        else:
            # Fallback to standard fused module
            return super()._create_fused_module(original_module, module_type)
    
    def _add_parallel_execution(self, model: nn.Module) -> nn.Module:
        """Add parallel execution capabilities to the model"""
        # Find independent operations that can run in parallel
        parallel_opportunities = self._find_parallel_opportunities(model)
        
        for module_path, parallel_modules in parallel_opportunities:
            # Replace sequential execution with parallel execution
            parent = self._get_parent_module(model, module_path)
            parallel_block = ParallelBlock(parallel_modules, self.stream_context)
            setattr(parent, module_path.split('.')[-1], parallel_block)
        
        return model
    
    def _find_parallel_opportunities(self, model: nn.Module) -> List[Tuple[str, List[nn.Module]]]:
        """Find modules that can be executed in parallel"""
        opportunities = []
        
        # Look for independent branches in the model
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList):
                # ModuleList often contains independent operations
                if len(module) > 1:
                    opportunities.append((name, list(module)))
            
            elif isinstance(module, nn.Sequential):
                # Look for independent operations in sequential blocks
                independent_ops = self._find_independent_operations(module)
                if len(independent_ops) > 1:
                    opportunities.append((name, independent_ops))
        
        return opportunities
    
    def _find_independent_operations(self, sequential: nn.Sequential) -> List[nn.Module]:
        """Find operations in a sequential block that could be parallelized"""
        # This is a simplified heuristic - in practice, would need dependency analysis
        independent = []
        
        # For now, just look for certain patterns that are typically independent
        for module in sequential:
            if isinstance(module, (StreamAwareFusedLinearReLU, StreamAwareFusedConv2dBNReLU)):
                independent.append(module)
        
        return independent if len(independent) > 1 else []


class StreamAwareInferenceWrapper:
    """Wrapper for stream-aware model inference"""
    
    def __init__(self, model: nn.Module, stream_context: StreamContext):
        self.model = model
        self.stream_context = stream_context
        self.inference_stream = stream_context.create_stream("inference", StreamPriority.HIGH)
    
    def __call__(self, *args, **kwargs):
        """Stream-aware inference call"""
        with self.stream_context.stream(self.inference_stream):
            return self.model(*args, **kwargs)
    
    def batch_inference(self, batches: List[torch.Tensor]) -> List[torch.Tensor]:
        """Parallel batch inference using multiple streams"""
        results = []
        num_streams = min(len(batches), 4)  # Limit to 4 streams
        
        # Create streams for batch processing
        batch_streams = [
            self.stream_context.create_stream(f"batch_{i}", StreamPriority.NORMAL)
            for i in range(num_streams)
        ]
        
        # Process batches in parallel
        for i, batch in enumerate(batches):
            stream = batch_streams[i % num_streams]
            
            with self.stream_context.stream(stream):
                with torch.no_grad():
                    result = self.model(batch)
                    results.append(result)
        
        # Synchronize all streams
        self.stream_context.synchronize()
        return results


def convert_model_with_streams(model: nn.Module,
                              input_shape: Optional[Tuple[int, ...]] = None,
                              device: str = 'cuda',
                              enable_parallel: bool = True,
                              validate_accuracy: bool = True) -> StreamAwareInferenceWrapper:
    """
    Convert model with stream support and return wrapped inference
    
    Args:
        model: PyTorch model to convert
        input_shape: Input shape for validation
        device: Target device
        enable_parallel: Enable parallel execution where possible
        validate_accuracy: Validate numerical accuracy
        
    Returns:
        StreamAwareInferenceWrapper for optimized inference
    """
    converter = StreamAwareModelConverter(
        validate_accuracy=validate_accuracy,
        accuracy_threshold=1e-5
    )
    
    converted_model, stream_context = converter.convert_model_with_streams(
        model, input_shape, device, enable_parallel
    )
    
    return StreamAwareInferenceWrapper(converted_model, stream_context)


# Convenience function for automatic stream-aware conversion
def auto_convert_with_streams(model: nn.Module,
                             input_shape: Optional[Tuple[int, ...]] = None,
                             device: str = 'cuda') -> StreamAwareInferenceWrapper:
    """
    One-line automatic conversion with stream support
    
    Example:
        >>> model = torchvision.models.resnet50(pretrained=True)
        >>> stream_model = auto_convert_with_streams(model, (1, 3, 224, 224))
        >>> 
        >>> # Single inference
        >>> output = stream_model(input_tensor)
        >>> 
        >>> # Parallel batch inference
        >>> outputs = stream_model.batch_inference([batch1, batch2, batch3])
    """
    return convert_model_with_streams(
        model, input_shape, device, 
        enable_parallel=True, 
        validate_accuracy=True
    )
