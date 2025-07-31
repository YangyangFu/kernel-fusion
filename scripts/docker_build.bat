@echo off
REM Docker Build and Management Scripts for Kernel Fusion Library (Windows)

setlocal enabledelayedexpansion

REM Function to print colored output (Windows equivalent)
set "INFO_PREFIX=[INFO]"
set "SUCCESS_PREFIX=[SUCCESS]"
set "WARNING_PREFIX=[WARNING]"
set "ERROR_PREFIX=[ERROR]"

REM Check if NVIDIA Docker runtime is available
:check_nvidia_docker
echo %INFO_PREFIX% Checking NVIDIA Docker runtime...
docker info | findstr nvidia >nul 2>&1
if errorlevel 1 (
    echo %WARNING_PREFIX% NVIDIA Docker runtime not detected. GPU support may not be available.
    exit /b 1
) else (
    echo %SUCCESS_PREFIX% NVIDIA Docker runtime detected
    exit /b 0
)

REM Build development image
:build_dev
echo %INFO_PREFIX% Building development image...
docker build -f Dockerfile.dev -t kernel-fusion:dev .
if errorlevel 1 (
    echo %ERROR_PREFIX% Failed to build development image
    exit /b 1
)
echo %SUCCESS_PREFIX% Development image built successfully
exit /b 0

REM Build production image
:build_prod
echo %INFO_PREFIX% Building production image...
docker build -f Dockerfile.prod -t kernel-fusion:latest .
if errorlevel 1 (
    echo %ERROR_PREFIX% Failed to build production image
    exit /b 1
)
echo %SUCCESS_PREFIX% Production image built successfully
exit /b 0

REM Build all images
:build_all
echo %INFO_PREFIX% Building all Docker images...
call :build_dev
if errorlevel 1 exit /b 1
call :build_prod
if errorlevel 1 exit /b 1
echo %SUCCESS_PREFIX% All images built successfully
exit /b 0

REM Start development environment
:start_dev
echo %INFO_PREFIX% Starting development environment...
docker-compose up -d kernel-fusion-dev
if errorlevel 1 (
    echo %ERROR_PREFIX% Failed to start development environment
    exit /b 1
)
echo %SUCCESS_PREFIX% Development environment started
echo %INFO_PREFIX% Access with: docker exec -it kernel-fusion-dev bash
exit /b 0

REM Start Jupyter notebook server
:start_jupyter
echo %INFO_PREFIX% Starting Jupyter notebook server...
docker-compose up -d kernel-fusion-jupyter
if errorlevel 1 (
    echo %ERROR_PREFIX% Failed to start Jupyter server
    exit /b 1
)
echo %SUCCESS_PREFIX% Jupyter server started
echo %INFO_PREFIX% Access at: http://localhost:8888
echo %INFO_PREFIX% Check logs for token: docker-compose logs kernel-fusion-jupyter
exit /b 0

REM Run tests
:run_tests
echo %INFO_PREFIX% Running tests...
docker-compose run --rm kernel-fusion-test
exit /b 0

REM Run benchmarks
:run_benchmarks
echo %INFO_PREFIX% Running benchmarks...
docker-compose run --rm kernel-fusion-benchmark
exit /b 0

REM Clean up containers and images
:cleanup
echo %INFO_PREFIX% Cleaning up Docker containers and images...
docker-compose down -v
docker image prune -f
echo %SUCCESS_PREFIX% Cleanup completed
exit /b 0

REM Show help
:show_help
echo Kernel Fusion Docker Management Script (Windows)
echo.
echo Usage: %~nx0 [COMMAND]
echo.
echo Commands:
echo   build-dev      Build development image
echo   build-prod     Build production image
echo   build-all      Build all images
echo   start-dev      Start development environment
echo   start-jupyter  Start Jupyter notebook server
echo   test           Run tests
echo   benchmark      Run benchmarks
echo   cleanup        Clean up containers and images
echo   help           Show this help message
echo.
echo Examples:
echo   %~nx0 build-all      # Build both dev and prod images
echo   %~nx0 start-dev      # Start development container
echo   %~nx0 test           # Run test suite
exit /b 0

REM Main script logic
if "%1"=="build-dev" (
    call :check_nvidia_docker
    if not errorlevel 1 call :build_dev
) else if "%1"=="build-prod" (
    call :check_nvidia_docker
    if not errorlevel 1 call :build_prod
) else if "%1"=="build-all" (
    call :check_nvidia_docker
    if not errorlevel 1 call :build_all
) else if "%1"=="start-dev" (
    call :check_nvidia_docker
    if not errorlevel 1 call :start_dev
) else if "%1"=="start-jupyter" (
    call :check_nvidia_docker
    if not errorlevel 1 call :start_jupyter
) else if "%1"=="test" (
    call :check_nvidia_docker
    if not errorlevel 1 call :run_tests
) else if "%1"=="benchmark" (
    call :check_nvidia_docker
    if not errorlevel 1 call :run_benchmarks
) else if "%1"=="cleanup" (
    call :cleanup
) else if "%1"=="help" (
    call :show_help
) else if "%1"=="--help" (
    call :show_help
) else if "%1"=="-h" (
    call :show_help
) else if "%1"=="" (
    call :show_help
) else (
    echo %ERROR_PREFIX% Unknown command: %1
    call :show_help
    exit /b 1
)
