// Optimized GPU code for 64000 x 20 x 4096 pointwise multiply using double-buffered streams with detailed timing and CSV logging
#include <cstdio>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>

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
    std::ofstream log("timing_report.csv");
    log << "buffer_id,H2D_ms,kernel_ms,D2H_ms\n";

    size_t chunk_bytes = CHUNK * sizeof(cuFloatComplex);

    cuFloatComplex *h_X[2], *h_H[2], *h_Y[2];
    for (int i = 0; i < 2; ++i) {
        cudaHostAlloc(&h_X[i], chunk_bytes, cudaHostAllocDefault);
        cudaHostAlloc(&h_H[i], chunk_bytes, cudaHostAllocDefault);
        cudaHostAlloc(&h_Y[i], chunk_bytes, cudaHostAllocDefault);
    }

    auto rand_X = randomComplexVector(TOTAL);
    auto rand_H = randomComplexVector(TOTAL);
    memcpy(h_X[0], rand_X.data(), chunk_bytes);
    memcpy(h_H[0], rand_H.data(), chunk_bytes);
    memcpy(h_X[1], rand_X.data() + CHUNK, chunk_bytes);
    memcpy(h_H[1], rand_H.data() + CHUNK, chunk_bytes);

    cuFloatComplex *d_X[2], *d_H[2], *d_Y[2];
    for (int i = 0; i < 2; ++i) {
        cudaMalloc(&d_X[i], chunk_bytes);
        cudaMalloc(&d_H[i], chunk_bytes);
        cudaMalloc(&d_Y[i], chunk_bytes);
    }

    cudaStream_t streams[2];
    cudaEvent_t start_H2D[2], stop_H2D[2], start_kernel[2], stop_kernel[2], start_D2H[2], stop_D2H[2];
    for (int i = 0; i < 2; ++i) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&start_H2D[i]);
        cudaEventCreate(&stop_H2D[i]);
        cudaEventCreate(&start_kernel[i]);
        cudaEventCreate(&stop_kernel[i]);
        cudaEventCreate(&start_D2H[i]);
        cudaEventCreate(&stop_D2H[i]);
    }

    auto wall_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 2; ++i) {
        cudaEventRecord(start_H2D[i], streams[i]);
        cudaMemcpyAsync(d_X[i], h_X[i], chunk_bytes, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_H[i], h_H[i], chunk_bytes, cudaMemcpyHostToDevice, streams[i]);
        cudaEventRecord(stop_H2D[i], streams[i]);

        cudaEventRecord(start_kernel[i], streams[i]);
        dim3 block(BLOCKSIZE);
        dim3 grid((CHUNK + block.x - 1) / block.x);
        flat_pw_multiply<<<grid, block, 0, streams[i]>>>(d_X[i], d_H[i], d_Y[i], CHUNK);
        cudaEventRecord(stop_kernel[i], streams[i]);

        cudaEventRecord(start_D2H[i], streams[i]);
        cudaMemcpyAsync(h_Y[i], d_Y[i], chunk_bytes, cudaMemcpyDeviceToHost, streams[i]);
        cudaEventRecord(stop_D2H[i], streams[i]);
    }

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    auto wall_end = std::chrono::high_resolution_clock::now();

    float h2d_ms[2], kernel_ms[2], d2h_ms[2];
    for (int i = 0; i < 2; ++i) {
        cudaEventElapsedTime(&h2d_ms[i], start_H2D[i], stop_H2D[i]);
        cudaEventElapsedTime(&kernel_ms[i], start_kernel[i], stop_kernel[i]);
        cudaEventElapsedTime(&d2h_ms[i], start_D2H[i], stop_D2H[i]);
        log << i << "," << h2d_ms[i] << "," << kernel_ms[i] << "," << d2h_ms[i] << "\n";
    }

    double wall_total_us = std::chrono::duration_cast<std::chrono::microseconds>(wall_end - wall_start).count();

    printf("Timing Report (per buffer):\n");
    for (int i = 0; i < 2; ++i) {
        printf("Buffer %d: H2D = %.3f ms, Kernel = %.3f ms, D2H = %.3f ms\n",
               i, h2d_ms[i], kernel_ms[i], d2h_ms[i]);
    }
    printf("Total wall-clock time: %.3f ms\n", wall_total_us / 1000.0);
    log << "Wall, , ," << wall_total_us / 1000.0 << "\n";
    log.close();

    for (int i = 0; i < 2; ++i) {
        cudaFree(d_X[i]); cudaFree(d_H[i]); cudaFree(d_Y[i]);
        cudaFreeHost(h_X[i]); cudaFreeHost(h_H[i]); cudaFreeHost(h_Y[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(start_H2D[i]); cudaEventDestroy(stop_H2D[i]);
        cudaEventDestroy(start_kernel[i]); cudaEventDestroy(stop_kernel[i]);
        cudaEventDestroy(start_D2H[i]); cudaEventDestroy(stop_D2H[i]);
    }

    return 0;
}
