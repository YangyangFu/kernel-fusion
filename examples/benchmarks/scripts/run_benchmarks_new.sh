#!/bin/bash
set -e

echo "=== Kernel Fusion Benchmark Suite ==="
echo ""

# Navigate to the benchmarks directory  
cd "$(dirname "$0")/.."

# Check if benchmarks are built
if [ ! -d "build" ]; then
    echo "❌ Build directory not found. Run ./scripts/build_with_pytorch.sh first"
    exit 1
fi

cd build

# Check which executables are available
AVAILABLE_BENCHMARKS=()
if [ -f "baseline_comparison" ]; then
    AVAILABLE_BENCHMARKS+=("baseline_comparison")
fi
if [ -f "elementwise_benchmark" ]; then
    AVAILABLE_BENCHMARKS+=("elementwise_benchmark")
fi
if [ -f "memory_benchmark" ]; then
    AVAILABLE_BENCHMARKS+=("memory_benchmark")
fi
if [ -f "comparison_benchmark" ]; then
    AVAILABLE_BENCHMARKS+=("comparison_benchmark")
fi
if [ -f "simple_fusion_validation" ]; then
    AVAILABLE_BENCHMARKS+=("simple_fusion_validation")
fi

if [ ${#AVAILABLE_BENCHMARKS[@]} -eq 0 ]; then
    echo "❌ No benchmark executables found. Build may have failed."
    exit 1
fi

echo "Available benchmarks: ${AVAILABLE_BENCHMARKS[*]}"
echo ""

# Parse command line arguments
RUN_MODE="all"
SPECIFIC_BENCHMARK=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --help, -h           Show this help message"
            echo "  --baseline           Run only baseline comparison (main benchmark)"
            echo "  --validation         Run only validation tests"
            echo "  --performance        Run only performance benchmarks"
            echo "  --list               List available benchmarks and exit"
            echo ""
            echo "Available benchmarks: ${AVAILABLE_BENCHMARKS[*]}"
            exit 0
            ;;
        --baseline)
            SPECIFIC_BENCHMARK="baseline_comparison"
            shift
            ;;
        --validation)
            RUN_MODE="validation"
            shift
            ;;
        --performance)
            RUN_MODE="performance"
            shift
            ;;
        --list)
            echo "Available benchmarks:"
            for bench in "${AVAILABLE_BENCHMARKS[@]}"; do
                echo "  - $bench"
            done
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to run a benchmark
run_benchmark() {
    local benchmark=$1
    if [[ " ${AVAILABLE_BENCHMARKS[*]} " =~ " $benchmark " ]]; then
        echo "--- Running $benchmark ---"
        ./$benchmark
        echo ""
    else
        echo "⚠️  $benchmark not available (not built)"
    fi
}

# Run benchmarks based on mode
if [ -n "$SPECIFIC_BENCHMARK" ]; then
    run_benchmark "$SPECIFIC_BENCHMARK"
elif [ "$RUN_MODE" = "validation" ]; then
    echo "=== Running Validation Tests ==="
    run_benchmark "simple_fusion_validation"
    if [ -f "fusion_validation" ]; then
        run_benchmark "fusion_validation"
    fi
elif [ "$RUN_MODE" = "performance" ]; then
    echo "=== Running Performance Benchmarks ==="
    run_benchmark "baseline_comparison"
    run_benchmark "elementwise_benchmark"
    run_benchmark "memory_benchmark"
    run_benchmark "comparison_benchmark"
else
    echo "=== Running All Available Benchmarks ==="
    echo ""
    
    # Run main benchmark first (most important)
    run_benchmark "baseline_comparison"
    
    # Run other performance benchmarks
    run_benchmark "elementwise_benchmark"
    run_benchmark "memory_benchmark" 
    run_benchmark "comparison_benchmark"
    
    # Run validation tests
    run_benchmark "simple_fusion_validation"
    if [ -f "fusion_validation" ]; then
        run_benchmark "fusion_validation"
    fi
fi

echo "=== Benchmark Suite Complete ==="
