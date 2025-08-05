#!/usr/bin/env python3
"""
KernelFusion Migration Script

This script helps migrate from the old PyTorch-only implementation
to the new standalone architecture while maintaining compatibility.

Usage:
    python migrate_to_standalone.py [--backup] [--keep-old]
    
Options:
    --backup    Create backup of old implementation
    --keep-old  Keep old implementation alongside new one
    --dry-run   Show what would be done without making changes
"""

import os
import shutil
import argparse
import subprocess
from pathlib import Path

def backup_old_implementation(workspace_root: Path, backup_dir: Path):
    """Create backup of old implementation"""
    print(f"📦 Creating backup in {backup_dir}")
    
    # Create backup directory
    backup_dir.mkdir(exist_ok=True)
    
    # Backup old kernel_fusion folder
    old_kf = workspace_root / "kernel_fusion"
    if old_kf.exists():
        shutil.copytree(old_kf, backup_dir / "kernel_fusion_old")
        print(f"   ✓ Backed up kernel_fusion/ to {backup_dir}/kernel_fusion_old/")
    
    # Backup old examples
    old_examples = workspace_root / "examples"
    if old_examples.exists():
        backup_examples = backup_dir / "examples_old"
        backup_examples.mkdir(exist_ok=True)
        
        for item in old_examples.iterdir():
            if item.name not in ["standalone", "pytorch_new"]:
                dest = backup_examples / item.name
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
        print(f"   ✓ Backed up old examples to {backup_dir}/examples_old/")

def create_new_structure(workspace_root: Path, keep_old: bool = False):
    """Create new directory structure"""
    print("🏗️  Creating new directory structure")
    
    # Create new directories
    new_dirs = [
        "frontends/pytorch",
        "frontends/tensorflow", 
        "frontends/jax",
        "frontends/c_api",
        "examples/pytorch",
        "examples/tensorflow",
        "examples/jax",
        "examples/benchmarks"
    ]
    
    for dir_path in new_dirs:
        full_path = workspace_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Created {dir_path}/")

def migrate_pytorch_frontend(workspace_root: Path):
    """Set up PyTorch frontend"""
    print("🐍 Setting up PyTorch frontend")
    
    # The frontend files were already created above
    frontend_dir = workspace_root / "frontends" / "pytorch"
    
    # Create setup.py for PyTorch frontend
    setup_py_content = '''
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

ext_modules = [
    Pybind11Extension(
        "kernel_fusion_pytorch",
        ["torch_bridge.cpp"],
        include_dirs=[
            "../../core/include",
        ],
        libraries=["kernel_fusion_core"],
        library_dirs=["../../build/core"],
        cxx_std=17,
    ),
]

setup(
    name="kernel_fusion_pytorch",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
'''
    
    setup_file = frontend_dir / "setup.py"
    with open(setup_file, 'w') as f:
        f.write(setup_py_content.strip())
    
    print(f"   ✓ Created {setup_file}")

def update_imports(workspace_root: Path, keep_old: bool = False):
    """Update import statements in existing code"""
    print("🔄 Updating import statements")
    
    # Create new kernel_fusion module that wraps the frontend
    new_kf_init = workspace_root / "kernel_fusion" / "__init__.py"
    
    if not keep_old:
        # Replace old __init__.py with frontend wrapper
        new_init_content = '''
# KernelFusion v2.0 - Standalone Architecture
# This module provides backward compatibility with the old interface
# while using the new standalone core library

try:
    # Try to import from new frontend
    from frontends.pytorch import *
    MIGRATION_STATUS = "complete"
except ImportError:
    try:
        # Fallback to old implementation if new frontend not available
        from .legacy_implementation import *
        MIGRATION_STATUS = "fallback"
        import warnings
        warnings.warn(
            "Using legacy KernelFusion implementation. "
            "Consider building the new standalone core library for better performance.",
            UserWarning
        )
    except ImportError:
        MIGRATION_STATUS = "failed"
        raise ImportError(
            "Neither new frontend nor legacy implementation available. "
            "Please check your KernelFusion installation."
        )

__version__ = "2.0.0-standalone"
'''
        
        # Backup current __init__.py as legacy
        if new_kf_init.exists():
            legacy_init = workspace_root / "kernel_fusion" / "legacy_implementation.py"
            shutil.copy2(new_kf_init, legacy_init)
            print(f"   ✓ Backed up old __init__.py as legacy_implementation.py")
        
        # Write new __init__.py
        with open(new_kf_init, 'w') as f:
            f.write(new_init_content.strip())
        print(f"   ✓ Updated {new_kf_init}")

def create_build_instructions(workspace_root: Path):
    """Create build instructions for new architecture"""
    print("📋 Creating build instructions")
    
    build_readme = workspace_root / "BUILD_STANDALONE.md"
    
    content = '''# Building KernelFusion Standalone

## Prerequisites

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compatible compiler
- Python 3.7+ (for Python frontends)

## Build Core Library

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release \\
         -DBUILD_PYTORCH_FRONTEND=ON \\
         -DBUILD_TENSORFLOW_FRONTEND=OFF \\
         -DBUILD_JAX_FRONTEND=OFF

# Build
make -j$(nproc)

# Install (optional)
sudo make install
```

## Build PyTorch Frontend

```bash
# Install PyTorch frontend
cd frontends/pytorch
pip install .
```

## Verify Installation

```bash
# Test C API
./build/examples/standalone/c_api_demo

# Test PyTorch frontend
python -c "import kernel_fusion; print(kernel_fusion.__version__)"
```

## Migration Status

Check migration status:
```python
import kernel_fusion
kernel_fusion.print_migration_status()
```
'''
    
    with open(build_readme, 'w') as f:
        f.write(content.strip())
    
    print(f"   ✓ Created {build_readme}")

def main():
    parser = argparse.ArgumentParser(description="Migrate to KernelFusion standalone architecture")
    parser.add_argument('--backup', action='store_true', help='Create backup of old implementation')
    parser.add_argument('--keep-old', action='store_true', help='Keep old implementation alongside new')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    
    args = parser.parse_args()
    
    # Get workspace root
    workspace_root = Path(__file__).parent
    
    print("🚀 KernelFusion Migration to Standalone Architecture")
    print("=" * 60)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print()
    
    # Show current structure
    print("📁 Current structure:")
    for item in workspace_root.iterdir():
        if item.is_dir():
            print(f"   📂 {item.name}/")
    print()
    
    if args.dry_run:
        print("Would perform the following actions:")
        print("1. Create backup (if --backup)")
        print("2. Create new directory structure")
        print("3. Set up PyTorch frontend")
        print("4. Update import statements")
        print("5. Create build instructions")
        return
    
    try:
        # Step 1: Backup if requested
        if args.backup:
            backup_dir = workspace_root / "backup"
            backup_old_implementation(workspace_root, backup_dir)
            print()
        
        # Step 2: Create new structure
        create_new_structure(workspace_root, args.keep_old)
        print()
        
        # Step 3: Set up PyTorch frontend
        migrate_pytorch_frontend(workspace_root)
        print()
        
        # Step 4: Update imports
        if not args.keep_old:
            update_imports(workspace_root, args.keep_old)
            print()
        
        # Step 5: Create build instructions
        create_build_instructions(workspace_root)
        print()
        
        print("✅ Migration completed successfully!")
        print()
        print("Next steps:")
        print("1. Build the core library: see BUILD_STANDALONE.md")
        print("2. Test the new implementation")
        print("3. Verify existing code still works")
        
        if args.keep_old:
            print("4. Remove old implementation when ready")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        if args.backup:
            print("Your original files are safely backed up in backup/")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
