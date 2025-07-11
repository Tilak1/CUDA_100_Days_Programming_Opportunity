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
constexpr size_t TOTAL = static_cast<size_t>(RAYS) * FRAMES * N;

constexpr int BLOCKSIZE = 256;
constexpr size_t CHUNK = TOTAL / 2; // Use two buffers

inline void gpuCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void flat_pw_multiply(const cuFloatComplex* X,
                                 const cuFloatComplex* H,
                                 cuFloatComplex* Y,
                                 size_t total) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < total) {
        Y[gid] = cuCmulf(X[gid], H[gid]);
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
    std::ofstream log("timing_report.csv");
    log << "buffer_id,H2D_ms,kernel_ms,D2H_ms,FLOPs,GFlops/s,BW_GBps,total_time_ms\n";

    size_t chunk_bytes = CHUNK * sizeof(cuFloatComplex);
    double flops_per_op = 6.0; // Complex multiply = 6 FLOPs
    double bytes_per_op = 3 * sizeof(cuFloatComplex); // 2 reads + 1 write

    cuFloatComplex *h_X[2], *h_H[2], *h_Y[2];
    for (int i = 0; i < 2; ++i) {
        gpuCheck(cudaHostAlloc(&h_X[i], chunk_bytes, cudaHostAllocDefault), "cudaHostAlloc h_X");
        gpuCheck(cudaHostAlloc(&h_H[i], chunk_bytes, cudaHostAllocDefault), "cudaHostAlloc h_H");
        gpuCheck(cudaHostAlloc(&h_Y[i], chunk_bytes, cudaHostAllocDefault), "cudaHostAlloc h_Y");
    }

    auto rand_X = randomComplexVector(TOTAL);
    auto rand_H = randomComplexVector(TOTAL);
    memcpy(h_X[0], rand_X.data(), chunk_bytes);
    memcpy(h_H[0], rand_H.data(), chunk_bytes);
    memcpy(h_X[1], rand_X.data() + CHUNK, chunk_bytes);
    memcpy(h_H[1], rand_H.data() + CHUNK, chunk_bytes);

    cuFloatComplex *d_X[2], *d_H[2], *d_Y[2];
    for (int i = 0; i < 2; ++i) {
        gpuCheck(cudaMalloc(&d_X[i], chunk_bytes), "cudaMalloc d_X");
        gpuCheck(cudaMalloc(&d_H[i], chunk_bytes), "cudaMalloc d_H");
        gpuCheck(cudaMalloc(&d_Y[i], chunk_bytes), "cudaMalloc d_Y");
    }

    cudaStream_t streams[2];
    cudaEvent_t start_H2D[2], stop_H2D[2], start_kernel[2], stop_kernel[2], start_D2H[2], stop_D2H[2];
    for (int i = 0; i < 2; ++i) {
        gpuCheck(cudaStreamCreate(&streams[i]), "cudaStreamCreate");
        gpuCheck(cudaEventCreate(&start_H2D[i]), "EventCreate start_H2D");
        gpuCheck(cudaEventCreate(&stop_H2D[i]), "EventCreate stop_H2D");
        gpuCheck(cudaEventCreate(&start_kernel[i]), "EventCreate start_kernel");
        gpuCheck(cudaEventCreate(&stop_kernel[i]), "EventCreate stop_kernel");
        gpuCheck(cudaEventCreate(&start_D2H[i]), "EventCreate start_D2H");
        gpuCheck(cudaEventCreate(&stop_D2H[i]), "EventCreate stop_D2H");
    }

    auto wall_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 2; ++i) {
        gpuCheck(cudaEventRecord(start_H2D[i], streams[i]), "EventRecord start_H2D");
        gpuCheck(cudaMemcpyAsync(d_X[i], h_X[i], chunk_bytes, cudaMemcpyHostToDevice, streams[i]), "MemcpyAsync d_X");
        gpuCheck(cudaMemcpyAsync(d_H[i], h_H[i], chunk_bytes, cudaMemcpyHostToDevice, streams[i]), "MemcpyAsync d_H");
        gpuCheck(cudaEventRecord(stop_H2D[i], streams[i]), "EventRecord stop_H2D");

        gpuCheck(cudaEventRecord(start_kernel[i], streams[i]), "EventRecord start_kernel");
        dim3 block(BLOCKSIZE);
        dim3 grid((CHUNK + block.x - 1) / block.x);
        flat_pw_multiply<<<grid, block, 0, streams[i]>>>(d_X[i], d_H[i], d_Y[i], CHUNK);
        gpuCheck(cudaGetLastError(), "Kernel Launch");
        gpuCheck(cudaEventRecord(stop_kernel[i], streams[i]), "EventRecord stop_kernel");

        gpuCheck(cudaEventRecord(start_D2H[i], streams[i]), "EventRecord start_D2H");
        gpuCheck(cudaMemcpyAsync(h_Y[i], d_Y[i], chunk_bytes, cudaMemcpyDeviceToHost, streams[i]), "MemcpyAsync d_Y");
        gpuCheck(cudaEventRecord(stop_D2H[i], streams[i]), "EventRecord stop_D2H");
    }

    for (int i = 0; i < 2; ++i)
        gpuCheck(cudaStreamSynchronize(streams[i]), "StreamSynchronize");

    auto wall_end = std::chrono::high_resolution_clock::now();

    float h2d_ms[2], kernel_ms[2], d2h_ms[2];
    for (int i = 0; i < 2; ++i) {
        gpuCheck(cudaEventElapsedTime(&h2d_ms[i], start_H2D[i], stop_H2D[i]), "ElapsedTime H2D");
        gpuCheck(cudaEventElapsedTime(&kernel_ms[i], start_kernel[i], stop_kernel[i]), "ElapsedTime kernel");
        gpuCheck(cudaEventElapsedTime(&d2h_ms[i], start_D2H[i], stop_D2H[i]), "ElapsedTime D2H");

        float total_time_ms = h2d_ms[i] + kernel_ms[i] + d2h_ms[i];
        double flops = CHUNK * flops_per_op;
        double gflops = flops / (kernel_ms[i] / 1000.0) / 1e9;
        double total_bytes = CHUNK * bytes_per_op;
        double gbps = total_bytes / (total_time_ms / 1000.0) / 1e9;

        log << i << "," << h2d_ms[i] << "," << kernel_ms[i] << "," << d2h_ms[i] << ","
            << flops << "," << gflops << "," << gbps << "," << total_time_ms << "\n";
    }

    double wall_total_us = std::chrono::duration_cast<std::chrono::microseconds>(wall_end - wall_start).count();

    printf("Timing Report (per buffer):\n");
    for (int i = 0; i < 2; ++i) {
        printf("Buffer %d: H2D = %.3f ms, Kernel = %.3f ms, D2H = %.3f ms\n",
               i, h2d_ms[i], kernel_ms[i], d2h_ms[i]);
    }
    printf("Total wall-clock time: %.3f ms\n", wall_total_us / 1000.0);
    log << "Wall, , , , , , ," << wall_total_us / 1000.0 << "\n";
    log.close();

    for (int i = 0; i < 2; ++i) {
        gpuCheck(cudaFree(d_X[i]), "Free d_X");
        gpuCheck(cudaFree(d_H[i]), "Free d_H");
        gpuCheck(cudaFree(d_Y[i]), "Free d_Y");
        gpuCheck(cudaFreeHost(h_X[i]), "FreeHost h_X");
        gpuCheck(cudaFreeHost(h_H[i]), "FreeHost h_H");
        gpuCheck(cudaFreeHost(h_Y[i]), "FreeHost h_Y");
        gpuCheck(cudaStreamDestroy(streams[i]), "Destroy stream");
        gpuCheck(cudaEventDestroy(start_H2D[i]), "Destroy event start_H2D");
        gpuCheck(cudaEventDestroy(stop_H2D[i]), "Destroy event stop_H2D");
        gpuCheck(cudaEventDestroy(start_kernel[i]), "Destroy event start_kernel");
        gpuCheck(cudaEventDestroy(stop_kernel[i]), "Destroy event stop_kernel");
        gpuCheck(cudaEventDestroy(start_D2H[i]), "Destroy event start_D2H");
        gpuCheck(cudaEventDestroy(stop_D2H[i]), "Destroy event stop_D2H");
    }

    return 0;
}
