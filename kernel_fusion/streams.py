#!/usr/bin/env python3
"""
Multi-Stream and Multi-Context Support for Kernel Fusion

This module implements CUDA stream management and multi-context execution
for parallel kernel execution and improved GPU utilization.

Features:
- CUDA stream management
- Context-aware operation execution
- Asynchronous kernel launches
- Stream synchronization primitives
- Memory pool management per stream
- Event-based synchronization
"""

import torch
import threading
from typing import Optional, Dict, List, Any, ContextManager
from contextlib import contextmanager
import weakref
from enum import Enum
import warnings

try:
    import kernel_fusion as kf
except ImportError:
    warnings.warn("kernel_fusion not available, using mock implementation")
    class MockKF:
        class ops:
            @staticmethod
            def add_relu(a, b, stream=None):
                return torch.relu(a + b)
    kf = MockKF()


class StreamPriority(Enum):
    """Stream priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2


class CUDAStreamManager:
    """Manages CUDA streams for parallel kernel execution"""
    
    def __init__(self):
        self._streams: Dict[str, torch.cuda.Stream] = {}
        self._default_stream = None
        self._stream_counter = 0
        self._lock = threading.Lock()
        
    def create_stream(self, name: Optional[str] = None, 
                     priority: StreamPriority = StreamPriority.NORMAL,
                     device: Optional[int] = None) -> torch.cuda.Stream:
        """Create a new CUDA stream"""
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, returning mock stream")
            return None
            
        with self._lock:
            if name is None:
                name = f"stream_{self._stream_counter}"
                self._stream_counter += 1
            
            if name in self._streams:
                return self._streams[name]
            
            # Create stream with priority
            if device is not None:
                with torch.cuda.device(device):
                    stream = torch.cuda.Stream(priority=priority.value)
            else:
                stream = torch.cuda.Stream(priority=priority.value)
            
            self._streams[name] = stream
            return stream
    
    def get_stream(self, name: str) -> Optional[torch.cuda.Stream]:
        """Get existing stream by name"""
        return self._streams.get(name)
    
    def destroy_stream(self, name: str):
        """Destroy a stream"""
        with self._lock:
            if name in self._streams:
                stream = self._streams[name]
                stream.synchronize()  # Wait for completion
                del self._streams[name]
    
    def synchronize_all(self):
        """Synchronize all streams"""
        for stream in self._streams.values():
            if stream is not None:
                stream.synchronize()
    
    def get_default_stream(self) -> torch.cuda.Stream:
        """Get the default stream"""
        if self._default_stream is None:
            self._default_stream = torch.cuda.default_stream()
        return self._default_stream
    
    def list_streams(self) -> List[str]:
        """List all active stream names"""
        return list(self._streams.keys())


class StreamContext:
    """Context manager for multi-stream operations"""
    
    def __init__(self, device: Optional[int] = None):
        self.device = device or torch.cuda.current_device() if torch.cuda.is_available() else 0
        self.stream_manager = CUDAStreamManager()
        self._active_stream = None
        self._events: Dict[str, torch.cuda.Event] = {}
        
    def create_stream(self, name: Optional[str] = None, 
                     priority: StreamPriority = StreamPriority.NORMAL) -> torch.cuda.Stream:
        """Create a new stream in this context"""
        return self.stream_manager.create_stream(name, priority, self.device)
    
    def create_event(self, name: str, enable_timing: bool = False) -> torch.cuda.Event:
        """Create a CUDA event for synchronization"""
        if not torch.cuda.is_available():
            return None
            
        event = torch.cuda.Event(enable_timing=enable_timing)
        self._events[name] = event
        return event
    
    def record_event(self, event_name: str, stream: Optional[torch.cuda.Stream] = None):
        """Record an event on a stream"""
        if event_name not in self._events:
            raise ValueError(f"Event '{event_name}' not found")
        
        event = self._events[event_name]
        if stream is None:
            stream = self._active_stream or torch.cuda.current_stream()
        
        if event is not None and stream is not None:
            event.record(stream)
    
    def wait_event(self, event_name: str, stream: Optional[torch.cuda.Stream] = None):
        """Wait for an event on a stream"""
        if event_name not in self._events:
            raise ValueError(f"Event '{event_name}' not found")
        
        event = self._events[event_name]
        if stream is None:
            stream = self._active_stream or torch.cuda.current_stream()
        
        if event is not None and stream is not None:
            stream.wait_event(event)
    
    @contextmanager
    def stream(self, stream: torch.cuda.Stream):
        """Context manager to execute operations on a specific stream"""
        if stream is None:
            yield
            return
            
        old_stream = self._active_stream
        self._active_stream = stream
        
        try:
            with torch.cuda.stream(stream):
                yield stream
        finally:
            self._active_stream = old_stream
    
    def synchronize(self):
        """Synchronize all streams in this context"""
        self.stream_manager.synchronize_all()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.synchronize()


class MultiStreamOperations:
    """Multi-stream aware operation implementations"""
    
    @staticmethod
    def add_relu_async(a: torch.Tensor, b: torch.Tensor, 
                      stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        """Asynchronous add + relu operation"""
        if stream is not None:
            with torch.cuda.stream(stream):
                return kf.ops.add_relu(a, b)
        else:
            return kf.ops.add_relu(a, b)
    
    @staticmethod
    def mul_tanh_async(a: torch.Tensor, b: torch.Tensor,
                      stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        """Asynchronous multiply + tanh operation"""
        if stream is not None:
            with torch.cuda.stream(stream):
                return kf.ops.mul_tanh(a, b)
        else:
            return kf.ops.mul_tanh(a, b)
    
    @staticmethod
    def linear_relu_async(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                         stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        """Asynchronous linear + relu operation"""
        if stream is not None:
            with torch.cuda.stream(stream):
                return kf.ops.fused_linear_relu(x, weight, bias)
        else:
            return kf.ops.fused_linear_relu(x, weight, bias)


class Pipeline:
    """Multi-stage pipeline with stream management"""
    
    def __init__(self, stages: List[callable], stream_context: StreamContext):
        self.stages = stages
        self.context = stream_context
        self.streams = []
        self.events = []
        
        # Create streams and events for each stage
        for i, stage in enumerate(stages):
            stream = stream_context.create_stream(f"pipeline_stage_{i}")
            event = stream_context.create_event(f"stage_{i}_complete", enable_timing=True)
            self.streams.append(stream)
            self.events.append(event)
    
    def execute(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Execute pipeline stages in parallel where possible"""
        results = inputs.copy()
        
        for i, (stage, stream, event) in enumerate(zip(self.stages, self.streams, self.events)):
            # Wait for previous stage if needed (data dependency)
            if i > 0:
                self.context.wait_event(f"stage_{i-1}_complete", stream)
            
            # Execute stage on its dedicated stream
            with self.context.stream(stream):
                results = stage(results)
                self.context.record_event(f"stage_{i}_complete", stream)
        
        # Wait for final stage
        self.context.wait_event(f"stage_{len(self.stages)-1}_complete")
        return results


class MemoryPool:
    """Stream-aware memory pool for efficient allocation"""
    
    def __init__(self):
        self._pools: Dict[torch.cuda.Stream, List[torch.Tensor]] = {}
        self._lock = threading.Lock()
    
    def get_tensor(self, shape: tuple, dtype: torch.dtype, 
                   stream: torch.cuda.Stream, device: int) -> torch.Tensor:
        """Get a tensor from the pool or allocate new one"""
        with self._lock:
            if stream not in self._pools:
                self._pools[stream] = []
            
            pool = self._pools[stream]
            
            # Look for reusable tensor
            for i, tensor in enumerate(pool):
                if (tensor.shape == shape and 
                    tensor.dtype == dtype and 
                    tensor.device.index == device):
                    return pool.pop(i)
            
            # Allocate new tensor
            return torch.empty(shape, dtype=dtype, device=device)
    
    def return_tensor(self, tensor: torch.Tensor, stream: torch.cuda.Stream):
        """Return tensor to the pool"""
        with self._lock:
            if stream not in self._pools:
                self._pools[stream] = []
            self._pools[stream].append(tensor)
    
    def clear_pool(self, stream: Optional[torch.cuda.Stream] = None):
        """Clear memory pool for a stream or all streams"""
        with self._lock:
            if stream is None:
                self._pools.clear()
            elif stream in self._pools:
                del self._pools[stream]


# Global instances
_default_stream_manager = CUDAStreamManager()
_default_memory_pool = MemoryPool()


def get_default_stream_manager() -> CUDAStreamManager:
    """Get the default global stream manager"""
    return _default_stream_manager


def get_default_memory_pool() -> MemoryPool:
    """Get the default global memory pool"""
    return _default_memory_pool


# Convenience functions
def create_stream_context(device: Optional[int] = None) -> StreamContext:
    """Create a new stream context"""
    return StreamContext(device)


@contextmanager
def parallel_streams(num_streams: int = 2, device: Optional[int] = None):
    """Context manager for parallel stream execution"""
    with create_stream_context(device) as ctx:
        streams = [ctx.create_stream(f"parallel_{i}") for i in range(num_streams)]
        yield ctx, streams


# Enhanced operation APIs with stream support
class ops:
    """Stream-aware operation APIs"""
    
    @staticmethod
    def add_relu(a: torch.Tensor, b: torch.Tensor, 
                stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        return MultiStreamOperations.add_relu_async(a, b, stream)
    
    @staticmethod
    def mul_tanh(a: torch.Tensor, b: torch.Tensor,
                stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        return MultiStreamOperations.mul_tanh_async(a, b, stream)
    
    @staticmethod
    def linear_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                   stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        return MultiStreamOperations.linear_relu_async(x, weight, bias, stream)
