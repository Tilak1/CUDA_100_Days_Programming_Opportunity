// Optimized CUDA code using pinned memory and streams for pointwise multiply benchmarking
#include <cstdio>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <random>
#include <vector>

constexpr int N = 4096;
constexpr int BLOCKSIZE = 128;
constexpr int BATCHES = 1000;

__global__ void pw_const_kernel(const cuFloatComplex* __restrict__ X,
                                cuFloatComplex*       __restrict__ Y,
                                int                   n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Y[i] = cuCmulf(X[i], Y[i]);
}

__constant__ cuFloatComplex H_const[N];

void gpuCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

std::vector<cuFloatComplex> randomComplexVector(int n) {
    std::mt19937 gen(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<cuFloatComplex> v(n);
    for (auto& c : v) c = make_cuFloatComplex(dist(gen), dist(gen));
    return v;
}

int main() {
    dim3 block(BLOCKSIZE);
    dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE);

    cuFloatComplex *d_X, *d_Y;
    cuFloatComplex *h_X_pinned, *h_Y_pinned;
    size_t bytes = N * sizeof(cuFloatComplex);

    // Allocate pinned host memory
    gpuCheck(cudaHostAlloc(&h_X_pinned, bytes, cudaHostAllocDefault), "host X pinned");
    gpuCheck(cudaHostAlloc(&h_Y_pinned, bytes, cudaHostAllocDefault), "host Y pinned");

    // Fill input data
    auto h_X_data = randomComplexVector(N);
    auto h_H_data = randomComplexVector(N);
    memcpy(h_X_pinned, h_X_data.data(), bytes);

    // Allocate device memory
    gpuCheck(cudaMalloc(&d_X, bytes), "malloc d_X");
    gpuCheck(cudaMalloc(&d_Y, bytes), "malloc d_Y");

    // Copy H to constant memory
    gpuCheck(cudaMemcpyToSymbol(H_const, h_H_data.data(), bytes), "const copy");

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Variables to accumulate timing data
    double total_h2d_time = 0.0;
    double total_kernel_time = 0.0;
    double total_d2h_time = 0.0;

    for (int b = 0; b < BATCHES; ++b) {
        cudaEvent_t start_h2d, stop_h2d, start_kernel, stop_kernel, start_d2h, stop_d2h;
        cudaEventCreate(&start_h2d);
        cudaEventCreate(&stop_h2d);
        cudaEventCreate(&start_kernel);
        cudaEventCreate(&stop_kernel);
        cudaEventCreate(&start_d2h);
        cudaEventCreate(&stop_d2h);

        // Async H2D
        cudaEventRecord(start_h2d, stream);
        cudaMemcpyAsync(d_X, h_X_pinned, bytes, cudaMemcpyHostToDevice, stream);
        cudaEventRecord(stop_h2d, stream);

        // Kernel
        cudaEventRecord(start_kernel, stream);
        pw_const_kernel<<<grid, block, 0, stream>>>(d_X, d_Y, N);
        cudaEventRecord(stop_kernel, stream);

        // Async D2H
        cudaEventRecord(start_d2h, stream);
        cudaMemcpyAsync(h_Y_pinned, d_Y, bytes, cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(stop_d2h, stream);

        // Sync and time
        cudaEventSynchronize(stop_d2h);

        float h2d_time = 0.f, kernel_time = 0.f, d2h_time = 0.f;
        cudaEventElapsedTime(&h2d_time, start_h2d, stop_h2d);
        cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
        cudaEventElapsedTime(&d2h_time, start_d2h, stop_d2h);

        // Accumulate times (convert ms to us)
        total_h2d_time += h2d_time * 1000.0;
        total_kernel_time += kernel_time * 1000.0;
        total_d2h_time += d2h_time * 1000.0;

        cudaEventDestroy(start_h2d);
        cudaEventDestroy(stop_h2d);
        cudaEventDestroy(start_kernel);
        cudaEventDestroy(stop_kernel);
        cudaEventDestroy(start_d2h);
        cudaEventDestroy(stop_d2h);
    }

    // Calculate and print averages
    double avg_h2d_time = total_h2d_time / BATCHES;
    double avg_kernel_time = total_kernel_time / BATCHES;
    double avg_d2h_time = total_d2h_time / BATCHES;

    //constexpr int BATCHES = 1000;
    printf("Benchmark Results (averaged over %d batches):\n", BATCHES);
    printf("Average H2D copy time: %.2f μs\n", avg_h2d_time);
    printf("Average kernel time: %.2f μs\n", avg_kernel_time);
    printf("Average D2H copy time: %.2f μs\n", avg_d2h_time);
    printf("Average total time per batch: %.2f μs\n", avg_h2d_time + avg_kernel_time + avg_d2h_time);

    cudaFree(d_X); 
    cudaFree(d_Y);
    cudaFreeHost(h_X_pinned); 
    cudaFreeHost(h_Y_pinned);
    cudaStreamDestroy(stream);

    return 0;
}