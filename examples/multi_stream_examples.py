#!/usr/bin/env python3
"""
Multi-Stream Examples for Kernel Fusion

This script demonstrates how to use multi-stream and multi-context support
for parallel kernel execution and improved GPU utilization.

Examples:
1. Basic parallel stream execution
2. Pipeline processing with multiple stages
3. Async model inference with streams
4. Memory pool management
5. Event-based synchronization
"""

import torch
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kernel_fusion.streams import (
    StreamContext, parallel_streams, Pipeline, 
    StreamPriority, ops as stream_ops
)


def example_basic_parallel_execution():
    """Example 1: Basic parallel stream execution"""
    print("🚀 Example 1: Basic Parallel Stream Execution")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cpu':
        print("⚠️  CUDA not available, multi-stream benefits won't be visible")
        return
    
    # Create test tensors
    a1 = torch.randn(1024, 1024, device=device)
    b1 = torch.randn(1024, 1024, device=device)
    a2 = torch.randn(1024, 1024, device=device)
    b2 = torch.randn(1024, 1024, device=device)
    
    # Sequential execution (baseline)
    torch.cuda.synchronize()
    start_time = time.time()
    
    result1_seq = stream_ops.add_relu(a1, b1)
    result2_seq = stream_ops.mul_tanh(a2, b2)
    
    torch.cuda.synchronize()
    sequential_time = time.time() - start_time
    
    # Parallel execution with multiple streams
    torch.cuda.synchronize()
    start_time = time.time()
    
    with parallel_streams(num_streams=2) as (ctx, streams):
        # Execute operations on different streams
        with ctx.stream(streams[0]):
            result1_par = stream_ops.add_relu(a1, b1, stream=streams[0])
        
        with ctx.stream(streams[1]):
            result2_par = stream_ops.mul_tanh(a2, b2, stream=streams[1])
        
        # Context automatically synchronizes all streams
    
    parallel_time = time.time() - start_time
    
    # Verify results are identical
    assert torch.allclose(result1_seq, result1_par, atol=1e-6)
    assert torch.allclose(result2_seq, result2_par, atol=1e-6)
    
    speedup = sequential_time / parallel_time
    print(f"Sequential time: {sequential_time*1000:.2f} ms")
    print(f"Parallel time:   {parallel_time*1000:.2f} ms")
    print(f"Speedup:         {speedup:.2f}x")
    print("✅ Results verified identical")


def example_pipeline_processing():
    """Example 2: Pipeline processing with multiple stages"""
    print("\n🔄 Example 2: Pipeline Processing")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("⚠️  CUDA not available, using CPU")
    
    # Define pipeline stages
    def stage1(inputs):
        """Stage 1: Element-wise operations"""
        x = inputs[0]
        return [stream_ops.add_relu(x, x)]
    
    def stage2(inputs):
        """Stage 2: More complex operations"""
        x = inputs[0]
        return [stream_ops.mul_tanh(x, x)]
    
    def stage3(inputs):
        """Stage 3: Final processing"""
        x = inputs[0]
        return [torch.sum(x, dim=-1)]
    
    # Create pipeline
    with StreamContext(device=0 if device == 'cuda' else None) as ctx:
        pipeline = Pipeline([stage1, stage2, stage3], ctx)
        
        # Input data
        input_tensor = torch.randn(512, 512, device=device)
        
        # Execute pipeline
        start_time = time.time()
        results = pipeline.execute([input_tensor])
        pipeline_time = time.time() - start_time
        
        print(f"Pipeline execution time: {pipeline_time*1000:.2f} ms")
        print(f"Final result shape: {results[0].shape}")
        print("✅ Pipeline completed successfully")


def example_async_model_inference():
    """Example 3: Async model inference with streams"""
    print("\n🧠 Example 3: Async Model Inference")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Simple model for demonstration
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(512, 256)
            self.linear2 = torch.nn.Linear(256, 128)
            self.linear3 = torch.nn.Linear(128, 10)
        
        def forward(self, x, stream=None):
            # Use stream-aware operations
            x = stream_ops.linear_relu(x, self.linear1.weight, self.linear1.bias, stream)
            x = stream_ops.linear_relu(x, self.linear2.weight, self.linear2.bias, stream)
            x = torch.nn.functional.linear(x, self.linear3.weight, self.linear3.bias)
            return x
    
    model = SimpleModel().to(device)
    model.eval()
    
    # Create multiple batches
    batches = [torch.randn(32, 512, device=device) for _ in range(4)]
    
    # Sequential inference
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    results_seq = []
    for batch in batches:
        with torch.no_grad():
            result = model(batch)
            results_seq.append(result)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    sequential_time = time.time() - start_time
    
    # Parallel inference with streams
    if device == 'cuda':
        torch.cuda.synchronize()
        start_time = time.time()
        
        results_par = []
        with StreamContext() as ctx:
            streams = [ctx.create_stream(f"inference_{i}") for i in range(2)]
            
            for i, batch in enumerate(batches):
                stream = streams[i % 2]  # Alternate between streams
                
                with ctx.stream(stream):
                    with torch.no_grad():
                        result = model(batch, stream=stream)
                        results_par.append(result)
        
        parallel_time = time.time() - start_time
        
        # Verify results
        for i, (seq, par) in enumerate(zip(results_seq, results_par)):
            assert torch.allclose(seq, par, atol=1e-5), f"Batch {i} results differ"
        
        speedup = sequential_time / parallel_time
        print(f"Sequential inference: {sequential_time*1000:.2f} ms")
        print(f"Parallel inference:   {parallel_time*1000:.2f} ms")
        print(f"Speedup:              {speedup:.2f}x")
        print("✅ All batches verified identical")
    else:
        print("CPU mode: parallel inference benefits not visible")


def example_event_synchronization():
    """Example 4: Event-based synchronization"""
    print("\n⏱️  Example 4: Event-based Synchronization")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("⚠️  CUDA not available, events not supported")
        return
    
    with StreamContext() as ctx:
        # Create streams
        stream1 = ctx.create_stream("producer")
        stream2 = ctx.create_stream("consumer")
        
        # Create events for timing
        start_event = ctx.create_event("start", enable_timing=True)
        middle_event = ctx.create_event("middle", enable_timing=True)
        end_event = ctx.create_event("end", enable_timing=True)
        
        # Producer-consumer pattern
        data = torch.randn(1024, 1024, device=device)
        
        # Producer stream
        with ctx.stream(stream1):
            ctx.record_event("start", stream1)
            intermediate = stream_ops.add_relu(data, data, stream=stream1)
            ctx.record_event("middle", stream1)
        
        # Consumer stream waits for producer
        with ctx.stream(stream2):
            ctx.wait_event("middle", stream2)  # Wait for producer
            final_result = stream_ops.mul_tanh(intermediate, intermediate, stream=stream2)
            ctx.record_event("end", stream2)
        
        # Synchronize and measure timing
        ctx.synchronize()
        
        # Calculate timings
        start_event_obj = ctx._events["start"]
        middle_event_obj = ctx._events["middle"]
        end_event_obj = ctx._events["end"]
        
        producer_time = start_event_obj.elapsed_time(middle_event_obj)
        consumer_time = middle_event_obj.elapsed_time(end_event_obj)
        total_time = start_event_obj.elapsed_time(end_event_obj)
        
        print(f"Producer time:  {producer_time:.2f} ms")
        print(f"Consumer time:  {consumer_time:.2f} ms")
        print(f"Total time:     {total_time:.2f} ms")
        print(f"Final result shape: {final_result.shape}")
        print("✅ Event synchronization completed")


def example_memory_pool():
    """Example 5: Memory pool management"""
    print("\n💾 Example 5: Memory Pool Management")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    from kernel_fusion.streams import get_default_memory_pool
    
    pool = get_default_memory_pool()
    
    with StreamContext() as ctx:
        stream = ctx.create_stream("memory_test")
        
        # Allocate tensors from pool
        shapes = [(100, 100), (200, 200), (100, 100)]  # Note: repeated shape
        tensors = []
        
        print("Allocating tensors from memory pool...")
        for i, shape in enumerate(shapes):
            tensor = pool.get_tensor(
                shape=shape,
                dtype=torch.float32,
                stream=stream,
                device=0 if device == 'cuda' else torch.device('cpu').index or 0
            )
            tensors.append(tensor)
            print(f"  Tensor {i}: {tensor.shape} at {tensor.data_ptr()}")
        
        # Use tensors
        with ctx.stream(stream):
            for tensor in tensors:
                tensor.fill_(1.0)
        
        ctx.synchronize()
        
        # Return tensors to pool
        print("Returning tensors to pool...")
        for i, tensor in enumerate(tensors):
            pool.return_tensor(tensor, stream)
            print(f"  Returned tensor {i}")
        
        # Allocate again (should reuse)
        print("Re-allocating tensors (should reuse)...")
        new_tensors = []
        for i, shape in enumerate(shapes):
            tensor = pool.get_tensor(
                shape=shape,
                dtype=torch.float32,
                stream=stream,
                device=0 if device == 'cuda' else torch.device('cpu').index or 0
            )
            new_tensors.append(tensor)
            print(f"  New tensor {i}: {tensor.shape} at {tensor.data_ptr()}")
        
        print("✅ Memory pool management completed")


def benchmark_multi_stream_performance():
    """Comprehensive performance benchmark"""
    print("\n📊 Multi-Stream Performance Benchmark")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("⚠️  CUDA not available, skipping performance benchmark")
        return
    
    # Test configurations
    configs = [
        {"name": "Small tensors", "shape": (256, 256), "ops": 10},
        {"name": "Medium tensors", "shape": (512, 512), "ops": 10},
        {"name": "Large tensors", "shape": (1024, 1024), "ops": 5},
    ]
    
    for config in configs:
        print(f"\n{config['name']} ({config['shape']})")
        print("-" * 30)
        
        # Generate test data
        tensors_a = [torch.randn(config['shape'], device=device) for _ in range(config['ops'])]
        tensors_b = [torch.randn(config['shape'], device=device) for _ in range(config['ops'])]
        
        # Sequential execution
        torch.cuda.synchronize()
        start_time = time.time()
        
        results_seq = []
        for a, b in zip(tensors_a, tensors_b):
            result = stream_ops.add_relu(a, b)
            results_seq.append(result)
        
        torch.cuda.synchronize()
        sequential_time = time.time() - start_time
        
        # Parallel execution with 2 streams
        torch.cuda.synchronize()
        start_time = time.time()
        
        results_par = []
        with parallel_streams(num_streams=2) as (ctx, streams):
            for i, (a, b) in enumerate(zip(tensors_a, tensors_b)):
                stream = streams[i % 2]
                with ctx.stream(stream):
                    result = stream_ops.add_relu(a, b, stream=stream)
                    results_par.append(result)
        
        parallel_time = time.time() - start_time
        
        # Parallel execution with 4 streams
        torch.cuda.synchronize()
        start_time = time.time()
        
        results_par4 = []
        with parallel_streams(num_streams=4) as (ctx, streams):
            for i, (a, b) in enumerate(zip(tensors_a, tensors_b)):
                stream = streams[i % 4]
                with ctx.stream(stream):
                    result = stream_ops.add_relu(a, b, stream=stream)
                    results_par4.append(result)
        
        parallel4_time = time.time() - start_time
        
        # Results
        speedup_2s = sequential_time / parallel_time
        speedup_4s = sequential_time / parallel4_time
        
        print(f"Sequential (1 stream): {sequential_time*1000:.2f} ms")
        print(f"Parallel (2 streams):  {parallel_time*1000:.2f} ms (speedup: {speedup_2s:.2f}x)")
        print(f"Parallel (4 streams):  {parallel4_time*1000:.2f} ms (speedup: {speedup_4s:.2f}x)")


if __name__ == "__main__":
    print("🌊 Kernel Fusion - Multi-Stream Examples")
    print("=" * 60)
    
    try:
        example_basic_parallel_execution()
        example_pipeline_processing()
        example_async_model_inference()
        example_event_synchronization()
        example_memory_pool()
        benchmark_multi_stream_performance()
        
        print("\n" + "=" * 60)
        print("✅ All multi-stream examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during examples: {e}")
        import traceback
        traceback.print_exc()
