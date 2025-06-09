// Optimized GPU code for 64000 x 20 x 4096 pointwise multiply using double-buffered streams
#include <cstdio>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <random>
#include <chrono>

constexpr int RAYS = 64000;
constexpr int FRAMES = 20;
constexpr int N = 4096; // FFT size
constexpr int TOTAL = RAYS * FRAMES * N;
constexpr int BLOCKSIZE = 256;
constexpr int CHUNK = TOTAL / 2; // Use two buffers

__global__ void flat_pw_multiply(const cuFloatComplex* X,
                                 const cuFloatComplex* H,
                                 cuFloatComplex* Y,
                                 int total) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total) {
        Y[gid] = cuCmulf(X[gid], H[gid]);
    }
}

std::vector<cuFloatComplex> randomComplexVector(int count) {
    std::mt19937 gen(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<cuFloatComplex> vec(count);
    for (auto& c : vec) c = make_cuFloatComplex(dist(gen), dist(gen));
    return vec;
}

int main() {
    size_t chunk_bytes = CHUNK * sizeof(cuFloatComplex);

    // Allocate host pinned memory (double buffer)
    cuFloatComplex *h_X[2], *h_H[2], *h_Y[2];
    for (int i = 0; i < 2; ++i) {
        cudaHostAlloc(&h_X[i], chunk_bytes, cudaHostAllocDefault);
        cudaHostAlloc(&h_H[i], chunk_bytes, cudaHostAllocDefault);
        cudaHostAlloc(&h_Y[i], chunk_bytes, cudaHostAllocDefault);
    }

    // Fill input with dummy data
    auto rand_X = randomComplexVector(TOTAL);
    auto rand_H = randomComplexVector(TOTAL);
    memcpy(h_X[0], rand_X.data(), chunk_bytes);
    memcpy(h_H[0], rand_H.data(), chunk_bytes);
    memcpy(h_X[1], rand_X.data() + CHUNK, chunk_bytes);
    memcpy(h_H[1], rand_H.data() + CHUNK, chunk_bytes);

    // Allocate device memory (double buffer)
    cuFloatComplex *d_X[2], *d_H[2], *d_Y[2];
    for (int i = 0; i < 2; ++i) {
        cudaMalloc(&d_X[i], chunk_bytes);
        cudaMalloc(&d_H[i], chunk_bytes);
        cudaMalloc(&d_Y[i], chunk_bytes);
    }

    // Create two streams
    cudaStream_t streams[2];
    for (int i = 0; i < 2; ++i) cudaStreamCreate(&streams[i]);

    // Timing start
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 2; ++i) {
        // Async H2D copy
        cudaMemcpyAsync(d_X[i], h_X[i], chunk_bytes, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_H[i], h_H[i], chunk_bytes, cudaMemcpyHostToDevice, streams[i]);

        // Kernel launch
        dim3 block(BLOCKSIZE);
        dim3 grid((CHUNK + block.x - 1) / block.x);
        flat_pw_multiply<<<grid, block, 0, streams[i]>>>(d_X[i], d_H[i], d_Y[i], CHUNK);

        // Async D2H copy
        cudaMemcpyAsync(h_Y[i], d_Y[i], chunk_bytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    // Sync both streams
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    auto end = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("Completed 64000x20x4096 in %.2f ms with double-buffered streams\n", total_us / 1000.0);

    // Cleanup
    for (int i = 0; i < 2; ++i) {
        cudaFree(d_X[i]); cudaFree(d_H[i]); cudaFree(d_Y[i]);
        cudaFreeHost(h_X[i]); cudaFreeHost(h_H[i]); cudaFreeHost(h_Y[i]);
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}
