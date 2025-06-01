#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 1000

__global__ void gemmGPUkernel(int *a, int *b, int *c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < n) && (col < n)) {
        int sum = 0;
        for (int k = 0; k < n; ++k)
            sum += a[row * n + k] * b[k * n + col];
            c[row * n + col] = sum;
        //c[row * n + col] = a[row * n + k] * b[k * n + col];
    }
}

void gemmCPUkernel(int *a, int *b, int *c, int n)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            c[i * n + j] = 0;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                c[i * n + j] += a[i * n + k] * b[k * n + j];
}

int main()
{
    int *d_a, *d_b, *d_c;

    // Host memory
    int *h_a = new int[N * N];
    int *h_b = new int[N * N];
    int *h_c = new int[N * N];
    int *h_cpu = new int[N * N];

    // Initialize inputs
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            h_a[i * N + j] = i;
            h_b[i * N + j] = N - i;
            h_c[i * N + j] = 0;
            h_cpu[i * N + j] = 0;
        }

    // Allocate GPU memory
    if (cudaMalloc((void**)&d_a, sizeof(int) * N * N) != cudaSuccess ||
        cudaMalloc((void**)&d_b, sizeof(int) * N * N) != cudaSuccess ||
        cudaMalloc((void**)&d_c, sizeof(int) * N * N) != cudaSuccess) {
        std::cerr << "CUDA malloc failed!" << std::endl;
        return -1;
    }

    // Copy input to device
    cudaMemcpy(d_a, h_a, sizeof(int) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * N * N, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Timing GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    gemmGPUkernel<<<grid, block>>>(d_a, d_b, d_c, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU execution time: " << milliseconds << " ms\n";

    // Timing CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    gemmCPUkernel(h_a, h_b, h_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU execution time: " << cpu_duration.count() << " ms\n";

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, sizeof(int) * N * N, cudaMemcpyDeviceToHost);

    // Validate result
    bool correct = true;
    for (int i = 0; i < N && correct; ++i)
        for (int j = 0; j < N; ++j) {
            if (h_c[i * N + j] != h_cpu[i * N + j]) {
                std::cerr << "Mismatch at (" << i << "," << j << "): "
                          << "GPU=" << h_c[i * N + j]
                          << ", CPU=" << h_cpu[i * N + j] << "\n";
                correct = false;
                break;
            }
        }

    std::cout << "Result match: " << (correct ? "YES" : "NO") << std::endl;

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_cpu;
    cudaDeviceReset();

    return 0;
}
