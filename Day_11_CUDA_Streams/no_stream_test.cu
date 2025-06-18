// no_streams.cu
#include <iostream>
#include <cuda_runtime.h>

__global__ void dummy_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        data[idx] = data[idx] * 2.0f;
}

int main() {
    int N = 1 << 24;
    size_t size = N * sizeof(float);
    float *h_data = new float[N];
    float *d_data;

    cudaMalloc(&d_data, size);

    for (int i = 0; i < N; ++i) h_data[i] = 1.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Blocking copy
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    dummy_kernel<<<(N + 255) / 256, 256>>>(d_data, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Time without streams: " << ms << " ms\n";

    cudaFree(d_data);
    delete[] h_data;
    return 0;
}
