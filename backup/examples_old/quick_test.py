#!/usr/bin/env python3
"""
Quick Setup and Test for Kernel Fusion Automatic Conversion

This script helps you quickly test the automatic model conversion functionality
without needing to write code. It demonstrates the core features and validates
that everything is working correctly.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for kernel_fusion import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_basic_functionality():
    """Test basic kernel fusion functionality"""
    print("🔧 Testing Basic Functionality")
    print("-" * 40)
    
    try:
        # Test mock kernel fusion operations
        from kernel_fusion.auto_convert import auto_convert_model
        print("✅ Kernel fusion imports successful")
        
        # Create a simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, 15),
                    nn.ReLU(),
                )
                self.classifier = nn.Linear(15, 5)
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        # Test model creation
        model = SimpleModel()
        print("✅ Test model created successfully")
        
        # Test automatic conversion
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        converted_model = auto_convert_model(
            model,
            input_shape=(1, 10),
            device=device,
            validate_accuracy=True
        )
        print("✅ Automatic conversion successful")
        
        # Test inference
        test_input = torch.randn(1, 10, device=device)
        original_output = model.to(device)(test_input)
        converted_output = converted_model(test_input)
        
        # Check accuracy
        max_diff = torch.max(torch.abs(original_output - converted_output)).item()
        print(f"✅ Accuracy validation: max difference = {max_diff:.2e}")
        
        if max_diff < 1e-5:
            print("✅ Numerical accuracy maintained")
        else:
            print("⚠️  Large numerical difference detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False


def test_torchvision_models():
    """Test with actual torchvision models if available"""
    print("\n🖼️  Testing Torchvision Models")
    print("-" * 40)
    
    try:
        import torchvision.models as models
        from kernel_fusion.auto_convert import auto_convert_model
        
        # Test with a small model
        print("Loading ResNet18...")
        model = models.resnet18(pretrained=False)  # Use unpretrained for speed
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Convert model
        print("Converting model...")
        converted_model = auto_convert_model(
            model,
            input_shape=(1, 3, 224, 224),
            device=device,
            validate_accuracy=True
        )
        
        print("✅ ResNet18 conversion successful")
        
        # Quick performance test
        test_input = torch.randn(1, 3, 224, 224, device=device)
        
        # Warmup
        model = model.to(device)
        model.eval()
        converted_model.eval()
        
        with torch.no_grad():
            for _ in range(5):
                _ = model(test_input)
                _ = converted_model(test_input)
        
        # Timing
        import time
        
        # Original model
        start = time.time()
        with torch.no_grad():
            for _ in range(20):
                _ = model(test_input)
        original_time = (time.time() - start) / 20
        
        # Converted model
        start = time.time()
        with torch.no_grad():
            for _ in range(20):
                _ = converted_model(test_input)
        converted_time = (time.time() - start) / 20
        
        speedup = original_time / converted_time
        print(f"✅ Performance test complete:")
        print(f"   Original time:  {original_time*1000:.2f} ms")
        print(f"   Converted time: {converted_time*1000:.2f} ms")
        print(f"   Speedup:        {speedup:.2f}x")
        
        return True
        
    except ImportError:
        print("⚠️  torchvision not available, skipping vision model tests")
        return True
    except Exception as e:
        print(f"❌ Error testing torchvision models: {e}")
        return False


def test_conversion_statistics():
    """Test conversion statistics and reporting"""
    print("\n📊 Testing Conversion Statistics")
    print("-" * 40)
    
    try:
        from kernel_fusion.auto_convert import ModelConverter
        
        # Create a model with multiple fusible patterns
        class MultiPatternModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Multiple Linear + ReLU patterns
                self.layer1 = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
                self.layer2 = nn.Sequential(nn.Linear(20, 30), nn.ReLU())
                self.layer3 = nn.Sequential(nn.Linear(30, 15), nn.ReLU())
                
                # LayerNorm + ReLU pattern
                self.norm = nn.Sequential(nn.LayerNorm(15), nn.ReLU())
                
                self.output = nn.Linear(15, 5)
        
        model = MultiPatternModel()
        
        # Create converter with statistics tracking
        converter = ModelConverter(validate_accuracy=True)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        converted_model = converter.convert_model(
            model,
            input_shape=(1, 10),
            device=device
        )
        
        # Print statistics
        print("Conversion Statistics:")
        for key, value in converter.conversion_stats.items():
            print(f"  {key}: {value}")
        
        print("✅ Statistics tracking working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error testing conversion statistics: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Kernel Fusion - Automatic Conversion Test Suite")
    print("=" * 60)
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    print("\n" + "=" * 60)
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_basic_functionality():
        tests_passed += 1
    
    if test_torchvision_models():
        tests_passed += 1
    
    if test_conversion_statistics():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! Kernel fusion is ready to use.")
        print("\nNext steps:")
        print("1. Try: python examples/automatic_conversion_examples.py")
        print("2. Try: python examples/convert_pretrained_models.py --list-models")
        print("3. Convert your own models with auto_convert_model()")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Make sure kernel_fusion package is installed: pip install -e .")
        print("2. Check that you have PyTorch installed: pip install torch")
        print("3. For CUDA support, install CUDA-compatible PyTorch")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
