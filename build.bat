@echo off
REM Build script for Windows

echo Building Kernel Fusion Library...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found in PATH
    exit /b 1
)

REM Check if CUDA is available
nvcc --version >nul 2>&1
if errorlevel 1 (
    echo Warning: NVCC not found. Building CPU-only version.
    set USE_CUDA=0
) else (
    echo CUDA toolkit found.
    set USE_CUDA=1
)

REM Clean previous builds
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
for /d %%i in (*.egg-info) do rmdir /s /q "%%i"

REM Install dependencies
echo Installing dependencies...
pip install torch numpy pybind11

REM Build the extension
echo Building extension...
if %USE_CUDA%==1 (
    echo Building with CUDA support...
    python setup.py build_ext --inplace
) else (
    echo Building CPU-only version...
    set CUDA_VISIBLE_DEVICES=""
    python setup.py build_ext --inplace
)

if errorlevel 1 (
    echo Error: Build failed
    exit /b 1
)

echo Build completed successfully!

REM Run basic test
echo Running basic test...
python -c "import kernel_fusion as kf; print('CUDA available:', kf.CUDA_AVAILABLE); print('Extension loaded:', kf.EXTENSION_LOADED)"

echo Done!
