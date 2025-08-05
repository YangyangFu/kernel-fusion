#!/usr/bin/env python3
"""
Pretrained Model Conversion Utilities

This script provides utilities for converting popular pretrained models
to use kernel fusion operations. Supports models from:
- torchvision (ResNet, DenseNet, EfficientNet, etc.)
- transformers library (BERT, GPT, T5, etc.)
- timm (Vision Transformers, etc.)

Usage:
    python convert_pretrained_models.py --model resnet50 --save-path ./models/
    python convert_pretrained_models.py --model bert-base-uncased --save-path ./models/
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
import argparse
import os
import sys
import json
from typing import Dict, Any, Optional, Tuple

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kernel_fusion.auto_convert import auto_convert_model

# Model registry with input shapes and load functions
MODEL_REGISTRY = {
    # Torchvision models
    'resnet18': {
        'loader': lambda: tv_models.resnet18(pretrained=True),
        'input_shape': (1, 3, 224, 224),
        'description': 'ResNet-18 from torchvision'
    },
    'resnet50': {
        'loader': lambda: tv_models.resnet50(pretrained=True),
        'input_shape': (1, 3, 224, 224),
        'description': 'ResNet-50 from torchvision'
    },
    'resnet101': {
        'loader': lambda: tv_models.resnet101(pretrained=True),
        'input_shape': (1, 3, 224, 224),
        'description': 'ResNet-101 from torchvision'
    },
    'densenet121': {
        'loader': lambda: tv_models.densenet121(pretrained=True),
        'input_shape': (1, 3, 224, 224),
        'description': 'DenseNet-121 from torchvision'
    },
    'vgg16': {
        'loader': lambda: tv_models.vgg16(pretrained=True),
        'input_shape': (1, 3, 224, 224),
        'description': 'VGG-16 from torchvision'
    },
    'mobilenet_v2': {
        'loader': lambda: tv_models.mobilenet_v2(pretrained=True),
        'input_shape': (1, 3, 224, 224),
        'description': 'MobileNet V2 from torchvision'
    },
    'efficientnet_b0': {
        'loader': lambda: tv_models.efficientnet_b0(pretrained=True),
        'input_shape': (1, 3, 224, 224),
        'description': 'EfficientNet-B0 from torchvision'
    }
}

# Add transformers models if available
try:
    from transformers import (
        BertModel, BertConfig,
        GPT2Model, GPT2Config,
        AutoModel, AutoConfig
    )
    
    TRANSFORMERS_MODELS = {
        'bert-base-uncased': {
            'loader': lambda: BertModel.from_pretrained('bert-base-uncased'),
            'input_shape': (1, 128),  # (batch_size, seq_len)
            'description': 'BERT Base Uncased from transformers',
            'input_type': 'token_ids'
        },
        'bert-large-uncased': {
            'loader': lambda: BertModel.from_pretrained('bert-large-uncased'),
            'input_shape': (1, 128),
            'description': 'BERT Large Uncased from transformers',
            'input_type': 'token_ids'
        },
        'gpt2': {
            'loader': lambda: GPT2Model.from_pretrained('gpt2'),
            'input_shape': (1, 100),
            'description': 'GPT-2 from transformers',
            'input_type': 'token_ids'
        }
    }
    
    MODEL_REGISTRY.update(TRANSFORMERS_MODELS)
    
except ImportError:
    print("transformers library not available. Skipping transformer models.")

# Add timm models if available
try:
    import timm
    
    TIMM_MODELS = {
        'vit_base_patch16_224': {
            'loader': lambda: timm.create_model('vit_base_patch16_224', pretrained=True),
            'input_shape': (1, 3, 224, 224),
            'description': 'Vision Transformer Base from timm'
        },
        'deit_small_patch16_224': {
            'loader': lambda: timm.create_model('deit_small_patch16_224', pretrained=True),
            'input_shape': (1, 3, 224, 224),
            'description': 'DeiT Small from timm'
        }
    }
    
    MODEL_REGISTRY.update(TIMM_MODELS)
    
except ImportError:
    print("timm library not available. Skipping timm models.")


def create_test_input(input_shape: Tuple[int, ...], input_type: str = 'float', device: str = 'cuda') -> torch.Tensor:
    """Create appropriate test input based on model requirements"""
    if input_type == 'token_ids':
        # For transformer models, create random token IDs
        vocab_size = 30522 if 'bert' in input_type else 50257  # BERT or GPT vocab size
        return torch.randint(0, vocab_size, input_shape, device=device)
    else:
        # For vision models, create random float tensors
        return torch.randn(input_shape, device=device)


def convert_and_save_model(model_name: str, 
                          save_path: str, 
                          device: str = 'cuda',
                          validate_accuracy: bool = True,
                          save_original: bool = False) -> Dict[str, Any]:
    """
    Convert a pretrained model and save it along with metadata
    
    Args:
        model_name: Name of the model from MODEL_REGISTRY
        save_path: Directory to save the converted model
        device: Device to use for conversion
        validate_accuracy: Whether to validate conversion accuracy
        save_original: Whether to also save the original model
        
    Returns:
        Dictionary with conversion results and metadata
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_info = MODEL_REGISTRY[model_name]
    
    print(f"Loading pretrained model: {model_name}")
    print(f"Description: {model_info['description']}")
    
    # Load the pretrained model
    try:
        original_model = model_info['loader']()
        original_model = original_model.to(device)
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return {'success': False, 'error': str(e)}
    
    # Create test input
    input_type = model_info.get('input_type', 'float')
    test_input = create_test_input(model_info['input_shape'], input_type, device)
    
    print(f"Input shape: {model_info['input_shape']}")
    print(f"Input type: {input_type}")
    
    # Convert the model
    print("Converting model to use fused kernels...")
    try:
        converted_model = auto_convert_model(
            original_model,
            input_shape=model_info['input_shape'],
            device=device,
            validate_accuracy=validate_accuracy,
            accuracy_threshold=1e-5
        )
    except Exception as e:
        print(f"Failed to convert model {model_name}: {e}")
        return {'success': False, 'error': str(e)}
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Save converted model
    converted_path = os.path.join(save_path, f"{model_name}_fused.pth")
    torch.save(converted_model.state_dict(), converted_path)
    print(f"Converted model saved to: {converted_path}")
    
    # Save original model if requested
    if save_original:
        original_path = os.path.join(save_path, f"{model_name}_original.pth")
        torch.save(original_model.state_dict(), original_path)
        print(f"Original model saved to: {original_path}")
    
    # Performance benchmark
    print("Benchmarking performance...")
    original_model.eval()
    converted_model.eval()
    
    def benchmark(model, input_tensor, num_runs=50):
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        import time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        return (time.time() - start_time) / num_runs * 1000  # ms
    
    original_time = benchmark(original_model, test_input)
    converted_time = benchmark(converted_model, test_input)
    speedup = original_time / converted_time
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'description': model_info['description'],
        'input_shape': model_info['input_shape'],
        'input_type': input_type,
        'device': device,
        'conversion_successful': True,
        'accuracy_validated': validate_accuracy,
        'performance': {
            'original_time_ms': round(original_time, 3),
            'converted_time_ms': round(converted_time, 3),
            'speedup': round(speedup, 3)
        },
        'files': {
            'converted_model': f"{model_name}_fused.pth",
            'original_model': f"{model_name}_original.pth" if save_original else None
        }
    }
    
    metadata_path = os.path.join(save_path, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_path}")
    
    # Print results
    print("\n" + "="*50)
    print("CONVERSION RESULTS")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Original inference time: {original_time:.2f} ms")
    print(f"Converted inference time: {converted_time:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print("="*50)
    
    return metadata


def convert_multiple_models(model_names: list, 
                           save_path: str, 
                           device: str = 'cuda',
                           validate_accuracy: bool = True,
                           save_original: bool = False) -> Dict[str, Any]:
    """Convert multiple models and save summary"""
    results = {}
    summary = {
        'total_models': len(model_names),
        'successful_conversions': 0,
        'failed_conversions': 0,
        'results': {}
    }
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Converting model {len(results)+1}/{len(model_names)}: {model_name}")
        print(f"{'='*60}")
        
        try:
            result = convert_and_save_model(
                model_name, save_path, device, validate_accuracy, save_original
            )
            
            if result.get('success', True):
                summary['successful_conversions'] += 1
                results[model_name] = result
            else:
                summary['failed_conversions'] += 1
                results[model_name] = result
                
        except Exception as e:
            print(f"Error converting {model_name}: {e}")
            summary['failed_conversions'] += 1
            results[model_name] = {'success': False, 'error': str(e)}
    
    summary['results'] = results
    
    # Save overall summary
    summary_path = os.path.join(save_path, "conversion_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total models: {summary['total_models']}")
    print(f"Successful: {summary['successful_conversions']}")
    print(f"Failed: {summary['failed_conversions']}")
    print(f"Summary saved to: {summary_path}")
    
    return summary


def list_available_models():
    """List all available models in the registry"""
    print("Available Models:")
    print("="*50)
    
    categories = {
        'Vision Models (torchvision)': [],
        'Transformer Models (transformers)': [],
        'Vision Models (timm)': []
    }
    
    for name, info in MODEL_REGISTRY.items():
        if 'torchvision' in info['description']:
            categories['Vision Models (torchvision)'].append((name, info['description']))
        elif 'transformers' in info['description']:
            categories['Transformer Models (transformers)'].append((name, info['description']))
        elif 'timm' in info['description']:
            categories['Vision Models (timm)'].append((name, info['description']))
        else:
            categories['Vision Models (torchvision)'].append((name, info['description']))
    
    for category, models in categories.items():
        if models:
            print(f"\n{category}:")
            print("-" * len(category))
            for name, desc in models:
                print(f"  {name:<25} - {desc}")


def main():
    parser = argparse.ArgumentParser(description='Convert pretrained models to use kernel fusion')
    parser.add_argument('--model', type=str, help='Model name to convert')
    parser.add_argument('--models', nargs='+', help='Multiple model names to convert')
    parser.add_argument('--save-path', type=str, default='./converted_models', 
                       help='Directory to save converted models')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use for conversion')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip accuracy validation')
    parser.add_argument('--save-original', action='store_true',
                       help='Also save original model weights')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    if not args.model and not args.models:
        print("Please specify --model or --models to convert, or --list-models to see available models")
        return
    
    validate_accuracy = not args.no_validate
    
    if args.model:
        # Convert single model
        convert_and_save_model(
            args.model, 
            args.save_path, 
            args.device, 
            validate_accuracy, 
            args.save_original
        )
    elif args.models:
        # Convert multiple models
        convert_multiple_models(
            args.models, 
            args.save_path, 
            args.device, 
            validate_accuracy, 
            args.save_original
        )


if __name__ == "__main__":
    main()
