#include <iostream>
#include <cuda_runtime.h>

#define Mask_width 5
#define BLOCK_SIZE 256

__constant__ float M[Mask_width];

// Declare shared memory arrays
__shared__ float S_A_PartialSHMEM[BLOCK_SIZE + Mask_width - 1];
__shared__ float S_A_FullSHMEM[BLOCK_SIZE + Mask_width - 1];


/* Below partial shared memory load kernel is not a good practice

The two step loading of data into shared memory is not a good practice
because it leads to warp divergence.

In this kernel, the threads in a warp are not all executing the same instructions at the same time.
This is because of the two-step loading of data into shared memory.

Tn-2 in every block loads data into shared memory. Tn-1 Last thread in the same block may not load data into shared memory becuase of offset condition 

And then some threads will go into above if and some will not
This is because of the condition that checks if offset is less than n_p
This is a partial shared memory load kernel

Ultimtaley leading to a warp divergence
This is not a good practice as it leads to warp divergence

*/

/* Second Kernel: Unrolls with Predication: 
i=0,4,8,12(12 will not be executed - out of bounds condition)
i=1,5,9,13(13 will not be executed - out of bounds condition)
i=2,6,10(10 will not be executed - out of bounds condition)
i=3,7,11(11 will not be executed - out of bounds condition)

We loaded from 0 to n_p into shared memory

No divergence: all threads loop over same structure

All the loops are unrolled and predicated i.e all indentical loops are made into separate instructions
This is a good practice as it leads to better performance
The number of loop iterations is small and predictable

So the compiler may unroll it. But in some cases, not all threads execute the same number of iterations (e.g., thread 0 does i = 0, 4, 8; thread 3 does i = 3, 7)
To avoid warp divergence, the compiler can use predication:
Converts if conditions into per-thread enable masks. All threads execute the same instructions, but only some write the result

So:
There’s no branch divergence Just masked execution (like a SIMD predicate register)

*/

__global__ void oned_convolution_kernel_warp_partialSHMEMLoad(const float* A, float* C, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;   
    
    // Calculate shared memory size needed
    int r = Mask_width / 2;
    int n_padded = blockDim.x + 2 * r;
    
    // Loading the first section of data into shared memory
    //if (tid < n + 2 * r) {  // Check bounds for padded array
        S_A_PartialSHMEM[threadIdx.x] = A[tid];
    //}

    // Loading second section of data into shared memory
    int offset = threadIdx.x + blockDim.x; // second half starting from blockDim.x
    int global_offset = blockIdx.x * blockDim.x + offset;
    
    if (offset < n_padded && global_offset < n + 2 * r) {
        S_A_PartialSHMEM[offset] = A[global_offset];
    }

    // Synchronize threads to ensure all data is loaded into shared memory
    __syncthreads();

    if (tid < n) {
        float result = 0.0f;
        
        // For partial SHMEM load, we need to adjust the indexing
        for (int j = 0; j < Mask_width; j++) {
            result += S_A_PartialSHMEM[threadIdx.x + j] * M[j];
        }
        
        C[tid] = result;
    }
}

__global__ void oned_convolution_kernel_loopUnrolling_lessWarpUsePreidcation_fullSHMEMLoad(const float* A, float* C, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate sizes
    int r = Mask_width / 2;
    int n_p = blockDim.x + 2 * r;  // Size for this block's shared memory
    
    // Calculate the starting position for this block (including left halo)
    int block_start = blockIdx.x * blockDim.x - r;
    
    // Load data into shared memory using loop unrolling
    for (int i = threadIdx.x; i < n_p; i += blockDim.x) {
        int global_idx = block_start + i;
        
        // Handle boundaries - pad with zeros
        if (global_idx >= 0 && global_idx < n) {
            S_A_FullSHMEM[i] = A[global_idx];
        } else {
            S_A_FullSHMEM[i] = 0.0f;
        }
    }
    
    __syncthreads();
    
    if (tid < n) {
        float result = 0.0f;
        
        // Convolution computation
        // threadIdx.x + r is the center position in shared memory
        for (int j = 0; j < Mask_width; j++) {
            result += S_A_FullSHMEM[threadIdx.x + r + j - r] * M[j];
            // Simplifies to: S_A_FullSHMEM[threadIdx.x + j] * M[j]
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
        int start = i - r;

        for (int j = 0; j < m; j++) {
            if ((start + j) >= 0 && (start + j) < n) {
                result[i] += A[start + j] * M[j];
            }
        }
        
        // Compare with GPU result
        if (abs(result[i] - C[i]) > 1e-5) {
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
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Using the full SHMEM load version
    oned_convolution_kernel_loopUnrolling_lessWarpUsePreidcation_fullSHMEMLoad<<<gridSize, blockSize>>>(d_A, d_C, n);
    checkCudaError("Kernel launch failed");
    
    // Copy result back
    cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify result
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