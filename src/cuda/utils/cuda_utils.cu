#include "cuda_utils.cuh"

LaunchConfig::LaunchConfig(int64_t total_elements, int block_size_x, cudaStream_t stream) 
    : stream(stream), shared_mem_size(0) {
    
    block_size = dim3(block_size_x, 1, 1);
    
    // Calculate grid size ensuring we cover all elements
    int64_t grid_size_x = (total_elements + block_size_x - 1) / block_size_x;
    
    // Limit grid size to maximum supported
    grid_size_x = std::min(grid_size_x, static_cast<int64_t>(65536));
    
    grid_size = dim3(static_cast<unsigned int>(grid_size_x), 1, 1);
}
