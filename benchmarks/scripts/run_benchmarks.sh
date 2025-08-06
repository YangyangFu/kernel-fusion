#!/bin/bash

# Comprehensive Kernel Fusion Benchmark Suite Runner
# Provides various modes for running benchmarks with detailed reporting

set -e

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

# Default settings
QUICK_MODE=false
VERBOSE=false
SAVE_RESULTS=false
OUTPUT_FILE=""

print_header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                Kernel Fusion Benchmark Suite                 ║${NC}"
    echo -e "${CYAN}║                                                              ║${NC}"
    echo -e "${CYAN}║  Comprehensive performance and validation testing for        ║${NC}"
    echo -e "${CYAN}║  kernel fusion implementations vs framework baselines       ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_usage() {
    echo -e "${YELLOW}Usage: $0 [OPTIONS] [BENCHMARK]${NC}"
    echo ""
    echo -e "${YELLOW}Modes:${NC}"
    echo "  --all                Run all available benchmarks (default)"
    echo "  --validation         Run only validation tests"
    echo "  --performance        Run only performance benchmarks"
    echo "  --baseline           Run only baseline comparison (main benchmark)"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  --quick              Use reduced iterations for faster execution"
    echo "  --verbose            Enable detailed output and intermediate results"
    echo "  --save-results FILE  Save benchmark results to specified file"
    echo "  --list               List available benchmarks and exit"
    echo "  --system-info        Show detailed system information"
    echo "  --help, -h           Show this help message"
    echo ""
    echo -e "${YELLOW}Individual Benchmarks:${NC}"
    echo "  baseline_comparison     Main performance comparison vs PyTorch/frameworks"
    echo "  simple_fusion_validation Basic correctness validation"
    echo "  elementwise_benchmark   Individual kernel performance analysis"
    echo "  memory_benchmark       Memory bandwidth analysis"
    echo "  comparison_benchmark   Direct kernel-to-kernel comparison"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0                              # Run all benchmarks"
    echo "  $0 --baseline --quick           # Quick baseline comparison"
    echo "  $0 --validation                 # Run only validation tests"
    echo "  $0 --save-results results.txt   # Save results to file"
    echo "  $0 baseline_comparison          # Run specific benchmark"
}

check_prerequisites() {
    echo -e "${BLUE}=== Checking Prerequisites ===${NC}"
    
    # Check if build directory exists
    if [ ! -d "$BUILD_DIR" ]; then
        echo -e "${RED}❌ Build directory not found: $BUILD_DIR${NC}"
        echo -e "${YELLOW}Please run the build script first:${NC}"
        echo "  ./scripts/build_with_pytorch.sh"
        exit 1
    fi
    
    # Check CUDA availability
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
        if [ "$VERBOSE" = true ]; then
            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        fi
    else
        echo -e "${YELLOW}⚠️  nvidia-smi not found - GPU status unknown${NC}"
    fi
    
    # Check PyTorch availability
    if python -c "import torch; print(f'PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>/dev/null; then
        echo -e "${GREEN}✅ PyTorch available for baseline comparisons${NC}"
    else
        echo -e "${YELLOW}⚠️  PyTorch not available - some benchmarks may be disabled${NC}"
    fi
    
    echo ""
}

discover_benchmarks() {
    cd "$BUILD_DIR"
    
    AVAILABLE_BENCHMARKS=()
    VALIDATION_BENCHMARKS=()
    PERFORMANCE_BENCHMARKS=()
    
    # Check for available executables
    local benchmarks=(
        "baseline_comparison:performance:Main performance comparison vs PyTorch/frameworks"
        "simple_fusion_validation:validation:Basic correctness validation without external deps"
        "elementwise_benchmark:performance:Individual elementwise kernel performance analysis"
        "memory_benchmark:performance:Memory bandwidth and access pattern analysis"
        "comparison_benchmark:performance:Direct kernel-to-kernel comparison benchmarks"
        "fusion_validation:validation:Comprehensive validation with PyTorch reference"
    )
    
    for benchmark_info in "${benchmarks[@]}"; do
        IFS=':' read -r name category description <<< "$benchmark_info"
        if [ -f "$name" ]; then
            AVAILABLE_BENCHMARKS+=("$name")
            if [ "$category" = "validation" ]; then
                VALIDATION_BENCHMARKS+=("$name")
            else
                PERFORMANCE_BENCHMARKS+=("$name")
            fi
            
            if [ "$VERBOSE" = true ]; then
                echo -e "${GREEN}✅ Found: $name${NC} - $description"
            fi
        else
            if [ "$VERBOSE" = true ]; then
                echo -e "${YELLOW}⚠️  Missing: $name${NC} - $description"
            fi
        fi
    done
    
    if [ ${#AVAILABLE_BENCHMARKS[@]} -eq 0 ]; then
        echo -e "${RED}❌ No benchmark executables found in $BUILD_DIR${NC}"
        echo -e "${YELLOW}Please build the project first:${NC}"
        echo "  ./scripts/build_with_pytorch.sh"
        exit 1
    fi
    
    echo -e "${GREEN}Found ${#AVAILABLE_BENCHMARKS[@]} available benchmarks${NC}"
    echo -e "${BLUE}Performance: ${#PERFORMANCE_BENCHMARKS[@]}, Validation: ${#VALIDATION_BENCHMARKS[@]}${NC}"
    echo ""
}

show_system_info() {
    echo -e "${BLUE}=== System Information ===${NC}"
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "User: $(whoami)"
    echo ""
    
    # CPU info
    echo -e "${YELLOW}CPU:${NC}"
    if command -v lscpu &> /dev/null; then
        lscpu | grep -E "Model name|CPU\(s\):|Thread|Core|MHz"
    else
        echo "CPU info not available"
    fi
    echo ""
    
    # Memory info
    echo -e "${YELLOW}Memory:${NC}"
    if command -v free &> /dev/null; then
        free -h
    else
        echo "Memory info not available"
    fi
    echo ""
    
    # GPU info
    echo -e "${YELLOW}GPU:${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv
    else
        echo "NVIDIA GPU info not available"
    fi
    echo ""
    
    # CUDA info
    echo -e "${YELLOW}CUDA:${NC}"
    if command -v nvcc &> /dev/null; then
        nvcc --version | grep -E "release|V"
    else
        echo "CUDA compiler not available"
    fi
    echo ""
    
    # PyTorch info
    echo -e "${YELLOW}PyTorch:${NC}"
    python -c "
import torch
print(f'Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>/dev/null || echo "PyTorch not available"
    echo ""
}

run_benchmark() {
    local benchmark=$1
    local description=$2
    
    if [[ ! " ${AVAILABLE_BENCHMARKS[*]} " =~ " $benchmark " ]]; then
        echo -e "${YELLOW}⚠️  Skipping $benchmark (not available)${NC}"
        return 1
    fi
    
    echo -e "${CYAN}━━━ Running: $benchmark ━━━${NC}"
    if [ -n "$description" ]; then
        echo -e "${BLUE}Description: $description${NC}"
    fi
    echo ""
    
    local start_time=$(date +%s)
    local benchmark_failed=false
    
    # Run the benchmark with timeout
    local timeout_duration=600  # 10 minutes
    local extra_args=""
    
    if [ "$QUICK_MODE" = true ] && [[ "$benchmark" != "simple_fusion_validation" ]]; then
        extra_args="--quick"
    fi
    
    if timeout $timeout_duration ./"$benchmark" $extra_args; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo ""
        echo -e "${GREEN}✅ $benchmark completed successfully in ${duration}s${NC}"
    else
        local exit_code=$?
        echo ""
        if [ $exit_code -eq 124 ]; then
            echo -e "${RED}❌ $benchmark timed out after ${timeout_duration}s${NC}"
        else
            echo -e "${RED}❌ $benchmark failed with exit code $exit_code${NC}"
        fi
        benchmark_failed=true
    fi
    
    echo ""
    if [ "$benchmark_failed" = true ]; then
        return 1
    else
        return 0
    fi
}

save_benchmark_results() {
    if [ "$SAVE_RESULTS" = true ] && [ -n "$OUTPUT_FILE" ]; then
        {
            echo "Kernel Fusion Benchmark Results"
            echo "Generated: $(date)"
            echo "System: $(hostname)"
            echo ""
            show_system_info
            echo ""
            echo "=== Benchmark Results ==="
        } > "$OUTPUT_FILE"
        echo -e "${GREEN}Results will be saved to: $OUTPUT_FILE${NC}"
    fi
}

# Parse command line arguments
RUN_MODE="all"
SPECIFIC_BENCHMARK=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            print_header
            print_usage
            exit 0
            ;;
        --all)
            RUN_MODE="all"
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
        --baseline)
            SPECIFIC_BENCHMARK="baseline_comparison"
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --save-results)
            SAVE_RESULTS=true
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --list)
            print_header
            check_prerequisites
            discover_benchmarks
            echo -e "${YELLOW}Available benchmarks:${NC}"
            for bench in "${AVAILABLE_BENCHMARKS[@]}"; do
                echo "  - $bench"
            done
            exit 0
            ;;
        --system-info)
            print_header
            show_system_info
            exit 0
            ;;
        baseline_comparison|simple_fusion_validation|elementwise_benchmark|memory_benchmark|comparison_benchmark|fusion_validation)
            SPECIFIC_BENCHMARK="$1"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_header
    
    if [ "$QUICK_MODE" = true ]; then
        echo -e "${YELLOW}🚀 Running in quick mode (reduced iterations)${NC}"
        echo ""
    fi
    
    check_prerequisites
    discover_benchmarks
    
    if [ "$SAVE_RESULTS" = true ]; then
        save_benchmark_results
    fi
    
    local failed_benchmarks=0
    local total_benchmarks=0
    
    echo -e "${BLUE}=== Starting Benchmark Execution ===${NC}"
    echo ""
    
    # Execute benchmarks based on mode
    if [ -n "$SPECIFIC_BENCHMARK" ]; then
        echo -e "${CYAN}Running specific benchmark: $SPECIFIC_BENCHMARK${NC}"
        echo ""
        total_benchmarks=1
        if ! run_benchmark "$SPECIFIC_BENCHMARK" ""; then
            failed_benchmarks=1
        fi
        
    elif [ "$RUN_MODE" = "validation" ]; then
        echo -e "${CYAN}Running validation benchmarks only${NC}"
        echo ""
        for benchmark in "${VALIDATION_BENCHMARKS[@]}"; do
            total_benchmarks=$((total_benchmarks + 1))
            case $benchmark in
                simple_fusion_validation)
                    if ! run_benchmark "$benchmark" "Basic correctness validation without external dependencies"; then
                        failed_benchmarks=$((failed_benchmarks + 1))
                    fi
                    ;;
                fusion_validation)
                    if ! run_benchmark "$benchmark" "Comprehensive validation with PyTorch reference implementations"; then
                        failed_benchmarks=$((failed_benchmarks + 1))
                    fi
                    ;;
            esac
        done
        
    elif [ "$RUN_MODE" = "performance" ]; then
        echo -e "${CYAN}Running performance benchmarks only${NC}"
        echo ""
        for benchmark in "${PERFORMANCE_BENCHMARKS[@]}"; do
            total_benchmarks=$((total_benchmarks + 1))
            case $benchmark in
                baseline_comparison)
                    if ! run_benchmark "$benchmark" "Main performance comparison vs PyTorch and other frameworks"; then
                        failed_benchmarks=$((failed_benchmarks + 1))
                    fi
                    ;;
                elementwise_benchmark)
                    if ! run_benchmark "$benchmark" "Individual elementwise kernel performance analysis"; then
                        failed_benchmarks=$((failed_benchmarks + 1))
                    fi
                    ;;
                memory_benchmark)
                    if ! run_benchmark "$benchmark" "Memory bandwidth and access pattern analysis"; then
                        failed_benchmarks=$((failed_benchmarks + 1))
                    fi
                    ;;
                comparison_benchmark)
                    if ! run_benchmark "$benchmark" "Direct kernel-to-kernel comparison benchmarks"; then
                        failed_benchmarks=$((failed_benchmarks + 1))
                    fi
                    ;;
            esac
        done
        
    else  # all mode
        echo -e "${CYAN}Running all available benchmarks${NC}"
        echo ""
        
        # Run main benchmark first (most important)
        if [[ " ${AVAILABLE_BENCHMARKS[*]} " =~ " baseline_comparison " ]]; then
            total_benchmarks=$((total_benchmarks + 1))
            if ! run_benchmark "baseline_comparison" "🎯 Main performance comparison vs PyTorch and other frameworks"; then
                failed_benchmarks=$((failed_benchmarks + 1))
            fi
        fi
        
        # Run validation tests
        for benchmark in "${VALIDATION_BENCHMARKS[@]}"; do
            total_benchmarks=$((total_benchmarks + 1))
            case $benchmark in
                simple_fusion_validation)
                    if ! run_benchmark "$benchmark" "✅ Basic correctness validation"; then
                        failed_benchmarks=$((failed_benchmarks + 1))
                    fi
                    ;;
                fusion_validation)
                    if ! run_benchmark "$benchmark" "✅ Comprehensive validation with PyTorch reference"; then
                        failed_benchmarks=$((failed_benchmarks + 1))
                    fi
                    ;;
            esac
        done
        
        # Run other performance benchmarks
        for benchmark in "${PERFORMANCE_BENCHMARKS[@]}"; do
            if [ "$benchmark" != "baseline_comparison" ]; then  # Already ran this
                total_benchmarks=$((total_benchmarks + 1))
                case $benchmark in
                    elementwise_benchmark)
                        if ! run_benchmark "$benchmark" "📊 Individual elementwise kernel performance analysis"; then
                            failed_benchmarks=$((failed_benchmarks + 1))
                        fi
                        ;;
                    memory_benchmark)
                        if ! run_benchmark "$benchmark" "💾 Memory bandwidth and access pattern analysis"; then
                            failed_benchmarks=$((failed_benchmarks + 1))
                        fi
                        ;;
                    comparison_benchmark)
                        if ! run_benchmark "$benchmark" "⚖️  Direct kernel-to-kernel comparison benchmarks"; then
                            failed_benchmarks=$((failed_benchmarks + 1))
                        fi
                        ;;
                esac
            fi
        done
    fi
    
    # Final summary
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                     Benchmark Summary                        ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    local success_count=$((total_benchmarks - failed_benchmarks))
    echo -e "${GREEN}✅ Successful: $success_count/$total_benchmarks${NC}"
    
    if [ $failed_benchmarks -gt 0 ]; then
        echo -e "${RED}❌ Failed: $failed_benchmarks/$total_benchmarks${NC}"
    fi
    
    echo ""
    echo "Completed at: $(date)"
    
    if [ "$SAVE_RESULTS" = true ] && [ -n "$OUTPUT_FILE" ]; then
        echo -e "${GREEN}Results saved to: $OUTPUT_FILE${NC}"
    fi
    
    # Additional information
    echo ""
    echo -e "${BLUE}💡 Pro Tips:${NC}"
    echo "  • Use --quick for faster execution during development"
    echo "  • Use --baseline to focus on the main performance comparison"
    echo "  • Use --verbose for detailed system information"
    echo "  • Use --save-results to capture output for reporting"
    echo ""
    echo -e "${YELLOW}For profiling and detailed analysis:${NC}"
    echo "  ncu --set full ./baseline_comparison    # Nsight Compute profiling"
    echo "  nsys profile ./baseline_comparison      # Nsight Systems tracing"
    
    # Return appropriate exit code
    if [ $failed_benchmarks -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

# Execute main function
main "$@"
