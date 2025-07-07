#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define Mask_width 5
#define BLOCK_SIZE 256

__constant__ float M[Mask_width];

__global__ void oned_convolution_kernel_proper(const float* A, float* C, int n) {
    // Declare shared memory inside kernel
    extern __shared__ float SHMEM[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int r = Mask_width / 2;  // radius
    
    // Calculate how much shared memory we need
    int n_padded = blockDim.x + 2 * r;
    
    // Load main data
    if (tid < n) {
        SHMEM[threadIdx.x + r] = A[tid];
    }
    
    // Load left halo
    if (threadIdx.x < r) {
        int left_idx = (blockIdx.x * blockDim.x) - r + threadIdx.x;
        if (left_idx >= 0 && left_idx < n) {
            SHMEM[threadIdx.x] = A[left_idx];
        } else {
            SHMEM[threadIdx.x] = 0.0f;  // Pad with zeros
        }
    }
    
    // Load right halo
    if (threadIdx.x < r) {
        int right_idx = (blockIdx.x * blockDim.x) + blockDim.x + threadIdx.x;
        if (right_idx < n) {
            SHMEM[blockDim.x + r + threadIdx.x] = A[right_idx];
        } else {
            SHMEM[blockDim.x + r + threadIdx.x] = 0.0f;  // Pad with zeros
        }
    }
    
    __syncthreads();
    
    // Compute convolution
    if (tid < n) {
        float result = 0.0f;
        
        // All data is now in shared memory
        for (int j = 0; j < Mask_width; j++) {
            result += SHMEM[threadIdx.x + j] * M[j];
        }
        
        C[tid] = result;
    }
}

// Alternative: Your approach but fixed
__global__ void oned_convolution_kernel_mixed_memory(const float* A, float* C, int n) {
    // Shared memory for main block data only
    __shared__ float SHMEM[BLOCK_SIZE];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load main data into shared memory
    if (tid < n) {
        SHMEM[threadIdx.x] = A[tid];
    }
    
    __syncthreads();
    
    if (tid < n) {
        float result = 0.0f;
        int r = Mask_width / 2;
        
        for (int j = 0; j < Mask_width; j++) {
            int idx = tid - r + j;  // Global index we need
            int local_idx = threadIdx.x - r + j;  // Local shared memory index
            
            // Check if we can use shared memory
            if (local_idx >= 0 && local_idx < blockDim.x && idx >= 0 && idx < n) {
                // Use shared memory
                result += SHMEM[local_idx] * M[j];
            } else if (idx >= 0 && idx < n) {
                // Fall back to global memory for halos
                result += A[idx] * M[j];
            }
            // else: out of bounds, contribute 0
        }
        
        C[tid] = result;
    }
}

// Host function to check for CUDA errors
void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << message << " - CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Corrected verify function
void verify_result(float* A, float* M, float* C, int n, int m) {
    int r = m / 2;
    float* result = new float[n];
    
    for (int i = 0; i < n; i++) {
        result[i] = 0.0f;
        
        for (int j = 0; j < m; j++) {
            int idx = i - r + j;
            if (idx >= 0 && idx < n) {
                result[i] += A[idx] * M[j];
            }
        }
        
        // Compare with GPU result
        if (fabs(result[i] - C[i]) > 1e-5) {
            std::cerr << "Mismatch at index " << i << ": CPU=" << result[i] 
                     << ", GPU=" << C[i] << std::endl;
        }
    }
    
    delete[] result;
}

// Example main function to demonstrate usage
int main() {
    int n = 1024;
    int mask_size = Mask_width;
    
    // Allocate host memory
    float *h_A = new float[n];
    float *h_C = new float[n];
    float *h_M = new float[mask_size];
    
    // Initialize data
    for (int i = 0; i < n; i++) {
        h_A[i] = i % 10;
    }
    for (int i = 0; i < mask_size; i++) {
        h_M[i] = 1.0f / mask_size;  // Simple averaging filter
    }
    
    // Allocate device memory
    float *d_A, *d_C;
    cudaMalloc(&d_A, n * sizeof(float));
    cudaMalloc(&d_C, n * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M, h_M, mask_size * sizeof(float));
    
    // Launch kernel
    int blockSize = BLOCK_SIZE;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Calculate shared memory size
    int r = Mask_width / 2;
    size_t shmem_size = (blockSize + 2 * r) * sizeof(float);
    
    // Using the proper shared memory version
    oned_convolution_kernel_proper<<<gridSize, blockSize, shmem_size>>>(d_A, d_C, n);
    checkCudaError("Kernel launch failed");
    
    // Copy result back
    cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify result
    verify_result(h_A, h_M, h_C, n, mask_size);
    
    std::cout << "Testing mixed memory version..." << std::endl;
    
    // Test the mixed memory version
    cudaMemset(d_C, 0, n * sizeof(float));
    oned_convolution_kernel_mixed_memory<<<gridSize, blockSize>>>(d_A, d_C, n);
    checkCudaError("Mixed memory kernel launch failed");
    
    cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost);
    verify_result(h_A, h_M, h_C, n, mask_size);
    
    // Cleanup
    delete[] h_A;
    delete[] h_C;
    delete[] h_M;
    cudaFree(d_A);
    cudaFree(d_C);
    
    std::cout << "Convolution completed successfully!" << std::endl;
    
    return 0;
}