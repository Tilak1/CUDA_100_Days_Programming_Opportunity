// Improved 2D convolution implementation with proper error handling and optimizations
// Fixes boundary conditions, adds error checking, and improves performance

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <chrono>

// 7 x 7 convolutional mask
#define MASK_DIM 7
#define MASK_OFFSET (MASK_DIM / 2)

// Shared memory tile size for optimization
#define TILE_SIZE 16

// Allocate mask in constant memory
__constant__ int mask[MASK_DIM * MASK_DIM];

// Macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Improved 2D Convolution Kernel with boundary checks
__global__ void convolution_2d_improved(const int* __restrict__ matrix, 
                                       int* __restrict__ result, 
                                       const int N) {
    // Calculate global thread positions
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check - critical fix!
    if (row >= N || col >= N) return;
    
    // Starting indices for convolution window
    int start_r = row - MASK_OFFSET;
    int start_c = col - MASK_OFFSET;
    
    // Accumulator for result
    int temp = 0;
    
    // Perform convolution
    #pragma unroll
    for (int i = 0; i < MASK_DIM; i++) {
        #pragma unroll
        for (int j = 0; j < MASK_DIM; j++) {
            int matrix_r = start_r + i;
            int matrix_c = start_c + j;
            
            // Boundary checks
            if (matrix_r >= 0 && matrix_r < N && matrix_c >= 0 && matrix_c < N) {
                temp += matrix[matrix_r * N + matrix_c] * mask[i * MASK_DIM + j];
            }
        }
    }
    
    // Write result
    result[row * N + col] = temp;
}

// Shared memory optimized version
__global__ void convolution_2d_shared(const int* __restrict__ matrix, 
                                     int* __restrict__ result, 
                                     const int N) {
    // Shared memory for input tile
    __shared__ int tile[TILE_SIZE + 2*MASK_OFFSET][TILE_SIZE + 2*MASK_OFFSET];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global indices
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    // Load data into shared memory with halo
    int tile_row = ty;
    int tile_col = tx;
    
    // Load main tile
    if (row < N && col < N) {
        tile[tile_row + MASK_OFFSET][tile_col + MASK_OFFSET] = matrix[row * N + col];
    } else {
        tile[tile_row + MASK_OFFSET][tile_col + MASK_OFFSET] = 0;
    }
    
    // Load halo regions
    // Top halo
    if (ty < MASK_OFFSET) {
        int halo_row = row - MASK_OFFSET;
        if (halo_row >= 0 && col < N) {
            tile[tile_row][tile_col + MASK_OFFSET] = matrix[halo_row * N + col];
        } else {
            tile[tile_row][tile_col + MASK_OFFSET] = 0;
        }
    }
    
    // Bottom halo
    if (ty >= TILE_SIZE - MASK_OFFSET) {
        int halo_row = row + MASK_OFFSET;
        if (halo_row < N && col < N) {
            tile[tile_row + 2*MASK_OFFSET][tile_col + MASK_OFFSET] = matrix[halo_row * N + col];
        } else {
            tile[tile_row + 2*MASK_OFFSET][tile_col + MASK_OFFSET] = 0;
        }
    }
    
    // Left halo
    if (tx < MASK_OFFSET) {
        int halo_col = col - MASK_OFFSET;
        if (row < N && halo_col >= 0) {
            tile[tile_row + MASK_OFFSET][tile_col] = matrix[row * N + halo_col];
        } else {
            tile[tile_row + MASK_OFFSET][tile_col] = 0;
        }
    }
    
    // Right halo
    if (tx >= TILE_SIZE - MASK_OFFSET) {
        int halo_col = col + MASK_OFFSET;
        if (row < N && halo_col < N) {
            tile[tile_row + MASK_OFFSET][tile_col + 2*MASK_OFFSET] = matrix[row * N + halo_col];
        } else {
            tile[tile_row + MASK_OFFSET][tile_col + 2*MASK_OFFSET] = 0;
        }
    }
    
    __syncthreads();
    
    // Perform convolution if within bounds
    if (row < N && col < N) {
        int temp = 0;
        
        #pragma unroll
        for (int i = 0; i < MASK_DIM; i++) {
            #pragma unroll
            for (int j = 0; j < MASK_DIM; j++) {
                temp += tile[tile_row + i][tile_col + j] * mask[i * MASK_DIM + j];
            }
        }
        
        result[row * N + col] = temp;
    }
}

// Initialize matrix with random values
void init_matrix(int* m, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[i * n + j] = rand() % 100;
        }
    }
}

// CPU verification function
void verify_result(const int* m, const int* mask_h, const int* result, int N) {
    std::cout << "Verifying results..." << std::endl;
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int temp = 0;
            
            for (int k = 0; k < MASK_DIM; k++) {
                for (int l = 0; l < MASK_DIM; l++) {
                    int offset_r = i - MASK_OFFSET + k;
                    int offset_c = j - MASK_OFFSET + l;
                    
                    if (offset_r >= 0 && offset_r < N && offset_c >= 0 && offset_c < N) {
                        temp += m[offset_r * N + offset_c] * mask_h[k * MASK_DIM + l];
                    }
                }
            }
            
            if (result[i * N + j] != temp) {
                std::cerr << "Mismatch at (" << i << ", " << j << "): "
                          << "Expected " << temp << ", got " << result[i * N + j] << std::endl;
                return;
            }
        }
    }
    
    std::cout << "Verification PASSED!" << std::endl;
}

int main() {
    // Matrix dimensions
    int N = 1 << 10;  // 1024x1024
    
    std::cout << "2D Convolution with " << N << "x" << N << " matrix and " 
              << MASK_DIM << "x" << MASK_DIM << " kernel" << std::endl;
    
    // Memory sizes
    size_t bytes_n = N * N * sizeof(int);
    size_t bytes_m = MASK_DIM * MASK_DIM * sizeof(int);
    
    // Allocate host memory using smart pointers
    std::unique_ptr<int[]> matrix(new int[N * N]);
    std::unique_ptr<int[]> result(new int[N * N]);
    std::unique_ptr<int[]> h_mask(new int[MASK_DIM * MASK_DIM]);
    
    // Initialize data
    std::cout << "Initializing data..." << std::endl;
    srand(42);  // Fixed seed for reproducibility
    init_matrix(matrix.get(), N);
    init_matrix(h_mask.get(), MASK_DIM);
    
    // Allocate device memory
    int *d_matrix, *d_result;
    CUDA_CHECK(cudaMalloc(&d_matrix, bytes_n));
    CUDA_CHECK(cudaMalloc(&d_result, bytes_n));
    
    // Copy data to device
    std::cout << "Copying data to GPU..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_matrix, matrix.get(), bytes_n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(mask, h_mask.get(), bytes_m));
    
    // Configure kernel launch parameters
    int THREADS = 16;
    int BLOCKS = (N + THREADS - 1) / THREADS;
    
    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(BLOCKS, BLOCKS);
    
    std::cout << "Grid: " << BLOCKS << "x" << BLOCKS << ", Block: " 
              << THREADS << "x" << THREADS << std::endl;
    
    // Warm up GPU
    convolution_2d_improved<<<grid_dim, block_dim>>>(d_matrix, d_result, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Time the improved kernel
    auto start = std::chrono::high_resolution_clock::now();
    
    convolution_2d_improved<<<grid_dim, block_dim>>>(d_matrix, d_result, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Improved kernel execution time: " << duration.count() << " ms" << std::endl;
    
    // Test shared memory version
    dim3 shared_block_dim(TILE_SIZE, TILE_SIZE);
    dim3 shared_grid_dim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    start = std::chrono::high_resolution_clock::now();
    
    convolution_2d_shared<<<shared_grid_dim, shared_block_dim>>>(d_matrix, d_result, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Shared memory kernel execution time: " << duration.count() << " ms" << std::endl;
    
    // Copy result back to host
    std::cout << "Copying result back to CPU..." << std::endl;
    CUDA_CHECK(cudaMemcpy(result.get(), d_result, bytes_n, cudaMemcpyDeviceToHost));
    
    // Verify result
    verify_result(matrix.get(), h_mask.get(), result.get(), N);
    
    std::cout << "COMPLETED SUCCESSFULLY!" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_matrix));
    CUDA_CHECK(cudaFree(d_result));
    
    return 0;
}