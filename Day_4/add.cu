# include <iostream>
# include <cuda_runtime.h>
#include <chrono>



__global__ void addGPUkernel(int *a, int *b, int *c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
    // Allocate memory on the GPU
    cudaMalloc((void**)&d_a, sizeof(int) * 10);
    cudaMalloc((void**)&d_b, sizeof(int) * 10);
    cudaMalloc((void**)&d_c, sizeof(int) * 10);

    int h_a[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int h_b[10] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    int h_c[10], h_cpu[10];
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, sizeof(int) * 10, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * 10, cudaMemcpyHostToDevice);
    // Launch kernel with 1 block of 10 threads

    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    addGPUkernel<<<1, 10>>>(d_a, d_b, d_c);
    cudaEventRecord(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error launching kernel: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaEventSynchronize(stop); // Now synchronizing after the event was recorded

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;


    // Measure CPU execution time

    auto cpu_start = std::chrono::high_resolution_clock::now();
    addCPUkernel(h_a, h_b, h_cpu, 10);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU execution time: " << cpu_duration.count() << " ms" << std::endl;



    // Copy result back to host
    cudaMemcpy(h_c, d_c, sizeof(int) * 10, cudaMemcpyDeviceToHost);


    // Print the result
    for (int i = 0; i < 10; i++) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // Return success
    cudaDeviceReset();
    std::cout << "Memory operations completed successfully." << std::endl;
    std::cout << "Exiting program." << std::endl;
    std::cout << "Program completed." << std::endl;



    return 0; 
}