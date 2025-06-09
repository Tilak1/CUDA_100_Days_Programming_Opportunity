// Optimized CUDA code using pinned memory and streams for pointwise multiply benchmarking
#include <cstdio>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <random>
#include <vector>
#include <fstream>

constexpr int N = 4096;
constexpr int BLOCKSIZE = 128;
constexpr int BATCHES = 1000;

__global__ void pw_const_kernel(const cuFloatComplex* __restrict__ X,
                                cuFloatComplex*       __restrict__ Y,
                                int                   n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Y[i] = cuCmulf(X[i], Y[i]);
}

__constant__ cuFloatComplex H_const[N];

void gpuCheck(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

std::vector<cuFloatComplex> randomComplexVector(int n)
{
    std::mt19937 gen(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<cuFloatComplex> v(n);
    for (auto& c : v) c = make_cuFloatComplex(dist(gen), dist(gen));
    return v;
}

int main()
{
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

    std::ofstream log("stream_timing_log.csv");
    log << "batch_id,copy_time_us,kernel_time_us\n";

    for (int b = 0; b < BATCHES; ++b) {
        cudaEvent_t start_copy, stop_copy, start_kernel, stop_kernel;
        cudaEventCreate(&start_copy);
        cudaEventCreate(&stop_copy);
        cudaEventCreate(&start_kernel);
        cudaEventCreate(&stop_kernel);

        // Async H2D
        cudaEventRecord(start_copy, stream);
        cudaMemcpyAsync(d_X, h_X_pinned, bytes, cudaMemcpyHostToDevice, stream);
        cudaEventRecord(stop_copy, stream);

        // Kernel
        cudaEventRecord(start_kernel, stream);
        pw_const_kernel<<<grid, block, 0, stream>>>(d_X, d_Y, N);
        cudaEventRecord(stop_kernel, stream);

        // Async D2H
        cudaMemcpyAsync(h_Y_pinned, d_Y, bytes, cudaMemcpyDeviceToHost, stream);

        // Sync and time
        cudaEventSynchronize(stop_kernel);

        float copy_time = 0.f, kernel_time = 0.f;
        cudaEventElapsedTime(&copy_time, start_copy, stop_copy);
        cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);

        log << b << "," << copy_time * 1000.0f << "," << kernel_time * 1000.0f << "\n";

        cudaEventDestroy(start_copy);
        cudaEventDestroy(stop_copy);
        cudaEventDestroy(start_kernel);
        cudaEventDestroy(stop_kernel);
    }

    log.close();
    cudaFree(d_X); cudaFree(d_Y);
    cudaFreeHost(h_X_pinned); cudaFreeHost(h_Y_pinned);
    cudaStreamDestroy(stream);

    printf("Done. Profile logged in stream_timing_log.csv\n");
    return 0;
}
