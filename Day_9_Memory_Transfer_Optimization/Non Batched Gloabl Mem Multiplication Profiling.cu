// Optimized GPU code for 64000 x 20 x 4096 pointwise multiply using pinned memory and stream
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
    size_t bytes = TOTAL * sizeof(cuFloatComplex);

    // Allocate host pinned memory
    cuFloatComplex *h_X_pinned, *h_H_pinned, *h_Y_pinned;
    cudaHostAlloc(&h_X_pinned, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_H_pinned, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_Y_pinned, bytes, cudaHostAllocDefault);

    // Fill input with dummy data
    auto rand_X = randomComplexVector(TOTAL);
    auto rand_H = randomComplexVector(TOTAL);
    memcpy(h_X_pinned, rand_X.data(), bytes);
    memcpy(h_H_pinned, rand_H.data(), bytes);

    // Allocate device memory
    cuFloatComplex *d_X, *d_H, *d_Y;
    cudaMalloc(&d_X, bytes);
    cudaMalloc(&d_H, bytes);
    cudaMalloc(&d_Y, bytes);

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Timing start
    auto start = std::chrono::high_resolution_clock::now();

    // Async H2D copy
    cudaMemcpyAsync(d_X, h_X_pinned, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_H, h_H_pinned, bytes, cudaMemcpyHostToDevice, stream);

    // Kernel launch
    dim3 block(BLOCKSIZE);
    dim3 grid((TOTAL + block.x - 1) / block.x);
    flat_pw_multiply<<<grid, block, 0, stream>>>(d_X, d_H, d_Y, TOTAL);

    // Async D2H copy
    cudaMemcpyAsync(h_Y_pinned, d_Y, bytes, cudaMemcpyDeviceToHost, stream);

    // Sync
    cudaStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("Completed 64000x20x4096 in %.2f ms\n", total_us / 1000.0);

    // Cleanup
    cudaFree(d_X); cudaFree(d_H); cudaFree(d_Y);
    cudaFreeHost(h_X_pinned); cudaFreeHost(h_H_pinned); cudaFreeHost(h_Y_pinned);
    cudaStreamDestroy(stream);

    return 0;
}
