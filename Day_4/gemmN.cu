#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 100000000
#define THREADS_PER_BLOCK 256

__global__ void addGPUkernel(int *a, int *b, int *c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

void addCPUkernel(int *a, int *b, int *c, int size)
{
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int *d_a, *d_b, *d_c;

    // Allocate memory on host
    int *h_a = new int[N];
    int *h_b = new int[N];
    int *h_c = new int[N];
    int *h_cpu = new int[N];

    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = N - i;
    }

    // Allocate memory on the GPU
    if (cudaMalloc((void**)&d_a, sizeof(int) * N) != cudaSuccess ||
        cudaMalloc((void**)&d_b, sizeof(int) * N) != cudaSuccess ||
        cudaMalloc((void**)&d_c, sizeof(int) * N) != cudaSuccess) {
        std::cerr << "CUDA malloc failed!" << std::endl;
        return -1;
    }

    // Copy data to device
    cudaMemcpy(d_a, h_a, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * N, cudaMemcpyHostToDevice);

    // Setup CUDA event timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    addGPUkernel<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    cudaEventRecord(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU kernel execution time: " << milliseconds << " ms" << std::endl;

    // CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    addCPUkernel(h_a, h_b, h_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU execution time: " << cpu_duration.count() << " ms" << std::endl;

    // Copy result back from GPU
    cudaMemcpy(h_c, d_c, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // Check result correctness (optional)
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_cpu[i]) {
            correct = false;
            std::cerr << "Mismatch at index " << i << ": GPU=" << h_c[i] << ", CPU=" << h_cpu[i] << std::endl;
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
