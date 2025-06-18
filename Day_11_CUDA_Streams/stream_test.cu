// multi_stream_overlap.cu
#include <iostream>
#include <cuda_runtime.h>

__global__ void dummy_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];
        for (int i = 0; i < 1000; ++i)
            val = val * 1.000001f;
        data[idx] = val;
    }
}

int main() {
    const int N = 1 << 24; // 16M elements per stream
    const size_t size = N * sizeof(float);
    const int num_streams = 2;

    float* h_data[num_streams];
    float* d_data[num_streams];
    cudaStream_t streams[num_streams];

    for (int i = 0; i < num_streams; ++i) {
        h_data[i] = new float[N];
        for (int j = 0; j < N; ++j) h_data[i][j] = 1.0f;

        cudaMalloc(&d_data[i], size);
        cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < num_streams; ++i) {
        cudaMemcpyAsync(d_data[i], h_data[i], size, cudaMemcpyHostToDevice, streams[i]);
        dummy_kernel<<<(N + 255) / 256, 256, 0, streams[i]>>>(d_data[i], N);
        cudaMemcpyAsync(h_data[i], d_data[i], size, cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Total time with " << num_streams << " streams: " << ms << " ms\n";

    for (int i = 0; i < num_streams; ++i) {
        cudaFree(d_data[i]);
        delete[] h_data[i];
        cudaStreamDestroy(streams[i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
