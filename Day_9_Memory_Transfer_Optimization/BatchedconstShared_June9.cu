#include <cstdio>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <random>
#include <chrono>

constexpr int RAYS = 64000;
constexpr int FRAMES = 20;
constexpr int N = 4096;
constexpr int BLOCKSIZE = 256;
constexpr int BATCH_SIZE = 1000; // Process 1000 rays at a time

inline void gpuCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void ray_pw_multiply_frame(const cuFloatComplex* X,
                                     const cuFloatComplex* H,
                                     cuFloatComplex* Y,
                                     int frame_id,
                                     int batch_rays) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int ray_id = blockIdx.y;
    
    if (ray_id < batch_rays && tid < N) {
        size_t x_base = static_cast<size_t>(ray_id) * FRAMES * N + frame_id * N;
        size_t h_base = frame_id * N;
        Y[x_base + tid] = cuCmulf(X[x_base + tid], H[h_base + tid]);
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
    const size_t batch_total = static_cast<size_t>(BATCH_SIZE) * FRAMES * N;
    const size_t h_total = FRAMES * N;
    
    printf("Processing %d rays in batches of %d\n", RAYS, BATCH_SIZE);
    printf("Memory per batch: %.2f GB\n", (2 * batch_total * sizeof(cuFloatComplex)) / 1e9);
    
    // Allocate memory for one batch
    cuFloatComplex *h_X_batch, *h_H, *h_Y_batch;
    h_X_batch = (cuFloatComplex*)malloc(batch_total * sizeof(cuFloatComplex));
    h_H = (cuFloatComplex*)malloc(h_total * sizeof(cuFloatComplex));
    h_Y_batch = (cuFloatComplex*)malloc(batch_total * sizeof(cuFloatComplex));
    
    if (!h_X_batch || !h_H || !h_Y_batch) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }
    
    // Generate H data once
    auto rand_H = randomComplexVector(h_total);
    memcpy(h_H, rand_H.data(), h_total * sizeof(cuFloatComplex));
    
    // Allocate GPU memory
    cuFloatComplex *d_X, *d_H, *d_Y;
    gpuCheck(cudaMalloc(&d_X, batch_total * sizeof(cuFloatComplex)), "cudaMalloc d_X");
    gpuCheck(cudaMalloc(&d_H, h_total * sizeof(cuFloatComplex)), "cudaMalloc d_H");
    gpuCheck(cudaMalloc(&d_Y, batch_total * sizeof(cuFloatComplex)), "cudaMalloc d_Y");
    
    // Copy H to GPU once
    gpuCheck(cudaMemcpy(d_H, h_H, h_total * sizeof(cuFloatComplex), cudaMemcpyHostToDevice), "cudaMemcpy d_H");
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    int num_batches = (RAYS + BATCH_SIZE - 1) / BATCH_SIZE;
    for (int batch = 0; batch < num_batches; ++batch) {
        int current_batch_size = std::min(BATCH_SIZE, RAYS - batch * BATCH_SIZE);
        size_t current_batch_total = static_cast<size_t>(current_batch_size) * FRAMES * N;
        
        // Generate random data for this batch
        auto rand_X_batch = randomComplexVector(current_batch_total);
        memcpy(h_X_batch, rand_X_batch.data(), current_batch_total * sizeof(cuFloatComplex));
        
        // Copy batch to GPU
        gpuCheck(cudaMemcpy(d_X, h_X_batch, current_batch_total * sizeof(cuFloatComplex), cudaMemcpyHostToDevice), "cudaMemcpy d_X batch");
        
        // Process all frames for this batch
        for (int f = 0; f < FRAMES; ++f) {
            dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, current_batch_size);
            dim3 block(BLOCKSIZE);
            ray_pw_multiply_frame<<<grid, block>>>(d_X, d_H, d_Y, f, current_batch_size);
            gpuCheck(cudaGetLastError(), "Kernel Launch");
        }
        
        gpuCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        
        // Copy results back (optional - comment out to skip verification)
        // gpuCheck(cudaMemcpy(h_Y_batch, d_Y, current_batch_total * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost), "cudaMemcpy h_Y batch");
        
        if (batch % 10 == 0) {
            printf("Processed batch %d/%d\n", batch + 1, num_batches);
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count() / 1000.0;
    
    printf("Batched processing completed in %.3f ms\n", elapsed_ms);
    printf("Processing rate: %.1f million complex multiplications per second\n", 
           (static_cast<double>(RAYS) * FRAMES * N / 1e6) / (elapsed_ms / 1000.0));
    
    // Cleanup
    cudaFree(d_X); cudaFree(d_H); cudaFree(d_Y);
    free(h_X_batch); free(h_H); free(h_Y_batch);
    
    return 0;
}