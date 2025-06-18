#include <cstdio>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <cstdlib>

constexpr int RAYS = 64000;
constexpr int FRAMES = 20;
constexpr int N = 4096; // FFT size
constexpr int BLOCKSIZE = 256; // Reduced to fit GPU limits

inline void gpuCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Option 1: Use global memory for H (simpler, still fast for A100)
__global__ void ray_pw_multiply_frame_v1(const cuFloatComplex* X,
                                         const cuFloatComplex* H,
                                         cuFloatComplex* Y,
                                         int frame_id) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int ray_id = blockIdx.y;
    
    if (tid < N) {
        size_t x_base = static_cast<size_t>(ray_id) * FRAMES * N + frame_id * N;
        size_t h_base = frame_id * N;
        Y[x_base + tid] = cuCmulf(X[x_base + tid], H[h_base + tid]);
    }
}

// Option 2: Use shared memory with proper tiling
__global__ void ray_pw_multiply_frame_v2(const cuFloatComplex* X,
                                         const cuFloatComplex* H,
                                         cuFloatComplex* Y,
                                         int frame_id) {
    __shared__ cuFloatComplex sh_H[BLOCKSIZE];
    
    int tid = threadIdx.x;
    int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int ray_id = blockIdx.y;
    
    // Load H into shared memory in tiles
    if (global_tid < N) {
        sh_H[tid] = H[frame_id * N + global_tid];
    }
    __syncthreads();
    
    if (global_tid < N) {
        size_t base = static_cast<size_t>(ray_id) * FRAMES * N + frame_id * N;
        Y[base + global_tid] = cuCmulf(X[base + global_tid], sh_H[tid]);
    }
}

std::vector<cuFloatComplex> randomComplexVector(size_t count) {
    std::mt19937 gen(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<cuFloatComplex> vec(count);
    for (auto& c : vec) c = make_cuFloatComplex(dist(gen), dist(gen));
    return vec;
}

int main() {
    const size_t total = static_cast<size_t>(RAYS) * FRAMES * N;
    const size_t h_total = FRAMES * N;
    
    // Check memory requirements
    size_t total_bytes = total * sizeof(cuFloatComplex);
    size_t h_bytes = h_total * sizeof(cuFloatComplex);
    printf("Memory requirements:\n");
    printf("  X data: %.2f GB\n", total_bytes / 1e9);
    printf("  H data: %.2f MB\n", h_bytes / 1e6);
    printf("  Y data: %.2f GB\n", total_bytes / 1e9);
    printf("  Total host memory needed: %.2f GB\n", (2 * total_bytes + h_bytes) / 1e9);
    
    // Use regular malloc instead of pinned memory to avoid allocation issues
    cuFloatComplex *h_X, *h_H, *h_Y;
    h_X = (cuFloatComplex*)malloc(total * sizeof(cuFloatComplex));
    h_H = (cuFloatComplex*)malloc(h_total * sizeof(cuFloatComplex));
    h_Y = (cuFloatComplex*)malloc(total * sizeof(cuFloatComplex));
    
    if (!h_X || !h_H || !h_Y) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }
    
    printf("Generating random data...\n");
    auto rand_X = randomComplexVector(total);
    auto rand_H = randomComplexVector(h_total);
    printf("Copying random data...\n");
    
    memcpy(h_X, rand_X.data(), total * sizeof(cuFloatComplex));
    memcpy(h_H, rand_H.data(), h_total * sizeof(cuFloatComplex));
    
    printf("Allocating GPU memory...\n");
    cuFloatComplex *d_X, *d_H, *d_Y;
    gpuCheck(cudaMalloc(&d_X, total * sizeof(cuFloatComplex)), "cudaMalloc d_X");
    gpuCheck(cudaMalloc(&d_H, h_total * sizeof(cuFloatComplex)), "cudaMalloc d_H");
    gpuCheck(cudaMalloc(&d_Y, total * sizeof(cuFloatComplex)), "cudaMalloc d_Y");
    
    printf("Copying data to GPU...\n");
    gpuCheck(cudaMemcpy(d_X, h_X, total * sizeof(cuFloatComplex), cudaMemcpyHostToDevice), "cudaMemcpy d_X");
    gpuCheck(cudaMemcpy(d_H, h_H, h_total * sizeof(cuFloatComplex), cudaMemcpyHostToDevice), "cudaMemcpy d_H");
    
    // Version 1: Simple global memory approach
    printf("Testing Version 1 (Global Memory):\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int f = 0; f < FRAMES; ++f) {
        dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, RAYS);
        dim3 block(BLOCKSIZE);
        ray_pw_multiply_frame_v1<<<grid, block>>>(d_X, d_H, d_Y, f);
        gpuCheck(cudaGetLastError(), "Kernel Launch V1");
    }
    gpuCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize V1");
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    printf("Global memory version completed in %.3f ms\n", elapsed_ms);
    
    // Version 2: Shared memory approach
    printf("\nTesting Version 2 (Shared Memory):\n");
    start = std::chrono::high_resolution_clock::now();
    
    for (int f = 0; f < FRAMES; ++f) {
        dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, RAYS);
        dim3 block(BLOCKSIZE);
        ray_pw_multiply_frame_v2<<<grid, block>>>(d_X, d_H, d_Y, f);
        gpuCheck(cudaGetLastError(), "Kernel Launch V2");
    }
    gpuCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize V2");
    
    end = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    printf("Shared memory version completed in %.3f ms\n", elapsed_ms);
    
    gpuCheck(cudaMemcpy(h_Y, d_Y, total * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost), "cudaMemcpy h_Y");
    
    // Memory bandwidth calculation
    size_t bytes_transferred = (2 * total + FRAMES * N) * sizeof(cuFloatComplex); // Read X, H, Write Y
    double bandwidth_gb_s = (bytes_transferred / 1e9) / (elapsed_ms / 1000.0);
    printf("Effective memory bandwidth: %.2f GB/s\n", bandwidth_gb_s);
    
    cudaFree(d_X); cudaFree(d_H); cudaFree(d_Y);
    free(h_X); free(h_H); free(h_Y);
    return 0;
}