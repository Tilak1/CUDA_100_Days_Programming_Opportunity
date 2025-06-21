#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

// Simple CUDA kernel for vector addition
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
        // Add some work to make the kernel take longer
    
        
    /*     for (int i = 0; i < 100; i++) {
            c[idx] = c[idx] * 1.001f;
        }
     */    
    
    }

}

// Check CUDA errors
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// Function to run without streams (synchronous)
void runWithoutStreams() {
    std::cout << "\n=== Running WITHOUT Streams (Synchronous) ===" << std::endl;
    
    const int N = 1024 * 1024;  // 1M elements
    const int bytes = N * sizeof(float);
    const int numOperations = 5;  // We'll do 4 separate operations
    
    // Host memory
    std::vector<float*> h_a(numOperations), h_b(numOperations), h_c(numOperations);
    // Device memory
    std::vector<float*> d_a(numOperations), d_b(numOperations), d_c(numOperations);
    
    // Allocate host memory (pinned for better transfer performance)
    for (int i = 0; i < numOperations; i++) {
        CHECK_CUDA(cudaMallocHost(&h_a[i], bytes));
        CHECK_CUDA(cudaMallocHost(&h_b[i], bytes));
        CHECK_CUDA(cudaMallocHost(&h_c[i], bytes));
        
        // Initialize data
        for (int j = 0; j < N; j++) {
            h_a[i][j] = 1.0f + i;
            h_b[i][j] = 2.0f + i;
        }
    }
    
    // Allocate device memory
    for (int i = 0; i < numOperations; i++) {
        CHECK_CUDA(cudaMalloc(&d_a[i], bytes));
        CHECK_CUDA(cudaMalloc(&d_b[i], bytes));
        CHECK_CUDA(cudaMalloc(&d_c[i], bytes));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Process each operation sequentially (no streams)
    for (int i = 0; i < numOperations; i++) {
        std::cout << "Processing operation " << i + 1 << std::endl;
        
        // Copy input data to device
        CHECK_CUDA(cudaMemcpy(d_a[i], h_a[i], bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b[i], h_b[i], bytes, cudaMemcpyHostToDevice));
        
        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a[i], d_b[i], d_c[i], N);
        
        // Copy result back to host
        CHECK_CUDA(cudaMemcpy(h_c[i], d_c[i], bytes, cudaMemcpyDeviceToHost));
        
        // Wait for completion (implicit with synchronous operations)
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Time without streams: " << duration.count() << " ms" << std::endl;
    
    // Verify results
    bool correct = true;

    for (int i = 0; i < numOperations; i++) {

        float expected = (1.0f + i) + (2.0f + i);  // a[i] + b[i]


        /* for (int iter = 0; iter < 100; iter++) {
            expected *= 1.001f;
        }
 */
        if (abs(h_c[i][0] - expected) > 0.01f) {
            correct = false;
            break;
        }

    }
    std::cout << "Results: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
    
    // Cleanup
    for (int i = 0; i < numOperations; i++) {
        cudaFreeHost(h_a[i]);
        cudaFreeHost(h_b[i]);
        cudaFreeHost(h_c[i]);
        cudaFree(d_a[i]);
        cudaFree(d_b[i]);
        cudaFree(d_c[i]);
    }
}

// Function to run with streams (asynchronous)
void runWithStreams() {
    std::cout << "\n=== Running WITH Streams (Asynchronous) ===" << std::endl;
    
    const int N = 1024 * 1024;  // 1M elements
    const int bytes = N * sizeof(float);
    const int numOperations = 5;  // We'll do 4 separate operations
    
    // Host memory
    std::vector<float*> h_a(numOperations), h_b(numOperations), h_c(numOperations);
    // Device memory
    std::vector<float*> d_a(numOperations), d_b(numOperations), d_c(numOperations);
    // CUDA streams
    std::vector<cudaStream_t> streams(numOperations);
    
    // Create streams
    for (int i = 0; i < numOperations; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }
    
    // Allocate host memory (pinned for async transfers)
    for (int i = 0; i < numOperations; i++) {
        CHECK_CUDA(cudaMallocHost(&h_a[i], bytes));
        CHECK_CUDA(cudaMallocHost(&h_b[i], bytes));
        CHECK_CUDA(cudaMallocHost(&h_c[i], bytes));
        
        // Initialize data
        for (int j = 0; j < N; j++) {
            h_a[i][j] = 1.0f + i;
            h_b[i][j] = 2.0f + i;
        }
    }
    
    // Allocate device memory
    for (int i = 0; i < numOperations; i++) {
        CHECK_CUDA(cudaMalloc(&d_a[i], bytes));
        CHECK_CUDA(cudaMalloc(&d_b[i], bytes));
        CHECK_CUDA(cudaMalloc(&d_c[i], bytes));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch all operations asynchronously using different streams
    for (int i = 0; i < numOperations; i++) {
        std::cout << "Launching operation " << i + 1 << " on stream " << i << std::endl;
        
        // Copy input data to device (async)
        CHECK_CUDA(cudaMemcpyAsync(d_a[i], h_a[i], bytes, cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_b[i], h_b[i], bytes, cudaMemcpyHostToDevice, streams[i]));
        
        // Launch kernel on specific stream
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_a[i], d_b[i], d_c[i], N);
        
        // Copy result back to host (async)
        CHECK_CUDA(cudaMemcpyAsync(h_c[i], d_c[i], bytes, cudaMemcpyDeviceToHost, streams[i]));
    }
    
    // Wait for all streams to complete
    for (int i = 0; i < numOperations; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Time with streams: " << duration.count() << " ms" << std::endl;
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < numOperations; i++) {
        float expected = (1.0f + i) + (2.0f + i);  // a[i] + b[i]
        
        /* for (int iter = 0; iter < 100; iter++) {
            expected *= 1.001f;
        }
 */

        if (abs(h_c[i][0] - expected) > 0.01f) {
            correct = false;
            break;
        }
    }
    std::cout << "Results: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
    
    // Cleanup
    for (int i = 0; i < numOperations; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFreeHost(h_a[i]);
        cudaFreeHost(h_b[i]);
        cudaFreeHost(h_c[i]);
        cudaFree(d_a[i]);
        cudaFree(d_b[i]);
        cudaFree(d_c[i]);
    }
}

// Function to demonstrate stream synchronization
void demonstrateStreamSynchronization() {
    std::cout << "\n=== Stream Synchronization Demo ===" << std::endl;
    
    cudaStream_t stream1, stream2;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));
    
    const int N = 1024 * 1024;
    const int bytes = N * sizeof(float);
    
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    
    // Allocate memory
    CHECK_CUDA(cudaMallocHost(&h_a, bytes));
    CHECK_CUDA(cudaMallocHost(&h_b, bytes));
    CHECK_CUDA(cudaMallocHost(&h_c, bytes));
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    std::cout << "Launching operations on different streams..." << std::endl;
    
    // Stream 1: Copy and compute
    CHECK_CUDA(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA(cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream1));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_a, d_b, d_c, N);
    
    // Stream 2: Independent operation (could be another kernel)
    std::cout << "Stream 1: Processing vector addition" << std::endl;
    std::cout << "Stream 2: Could be processing other work simultaneously" << std::endl;
    
    // Copy result back on stream 1
    CHECK_CUDA(cudaMemcpyAsync(h_c, d_c, bytes, cudaMemcpyDeviceToHost, stream1));
    
    // Wait for specific stream
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    std::cout << "Stream 1 completed!" << std::endl;
    
    // Cleanup
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    std::cout << "CUDA Streams Demonstration" << std::endl;
    std::cout << "=========================" << std::endl;
    
    // Check CUDA device
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Max concurrent streams: " << prop.asyncEngineCount << std::endl;
    
    // Run demonstrations
    runWithoutStreams();
    runWithStreams();
    demonstrateStreamSynchronization();
    
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    std::cout << "The stream version should be faster because:" << std::endl;
    std::cout << "1. Memory transfers and kernel execution overlap" << std::endl;
    std::cout << "2. Multiple operations run concurrently" << std::endl;
    std::cout << "3. Better GPU resource utilization" << std::endl;
    
    return 0;
}