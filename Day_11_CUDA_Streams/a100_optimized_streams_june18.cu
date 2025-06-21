#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

// More complex kernel to better utilize A100's compute power
__global__ void complexVectorOp(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float temp = a[idx] + b[idx];
        
        // More compute-intensive work to utilize A100's power
        for (int i = 0; i < 500; i++) {
            temp = temp * 1.001f + sinf(temp * 0.01f);
            temp = sqrtf(fabsf(temp));
        }
        
        c[idx] = temp;
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

void runWithOptimalStreams(int numStreams) {
    std::cout << "\n=== Running with " << numStreams << " streams (A100 Optimized) ===" << std::endl;
    
    const int N = 2 * 1024 * 1024;  // 2M elements (larger for A100)
    const int bytes = N * sizeof(float);
    
    // Host memory vectors
    std::vector<float*> h_a(numStreams), h_b(numStreams), h_c(numStreams);
    // Device memory vectors
    std::vector<float*> d_a(numStreams), d_b(numStreams), d_c(numStreams);
    // CUDA streams
    std::vector<cudaStream_t> streams(numStreams);
    
    // Create streams
    for (int i = 0; i < numStreams; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }
    
    // Allocate pinned host memory for better transfer performance
    for (int i = 0; i < numStreams; i++) {
        CHECK_CUDA(cudaMallocHost(&h_a[i], bytes));
        CHECK_CUDA(cudaMallocHost(&h_b[i], bytes));
        CHECK_CUDA(cudaMallocHost(&h_c[i], bytes));
        
        // Initialize with different values per stream
        for (int j = 0; j < N; j++) {
            h_a[i][j] = 1.0f + i * 0.1f + j * 0.001f;
            h_b[i][j] = 2.0f + i * 0.1f + j * 0.001f;
        }
    }
    
    // Allocate device memory
    for (int i = 0; i < numStreams; i++) {
        CHECK_CUDA(cudaMalloc(&d_a[i], bytes));
        CHECK_CUDA(cudaMalloc(&d_b[i], bytes));
        CHECK_CUDA(cudaMalloc(&d_c[i], bytes));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch all operations asynchronously
    std::cout << "Launching " << numStreams << " concurrent operations..." << std::endl;
    for (int i = 0; i < numStreams; i++) {
        // H2D transfers
        CHECK_CUDA(cudaMemcpyAsync(d_a[i], h_a[i], bytes, cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_b[i], h_b[i], bytes, cudaMemcpyHostToDevice, streams[i]));
        
        // Kernel launch - optimized for A100
        int threadsPerBlock = 512;  // A100 can handle larger blocks efficiently
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        complexVectorOp<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_a[i], d_b[i], d_c[i], N);
        
        // D2H transfer  
        CHECK_CUDA(cudaMemcpyAsync(h_c[i], d_c[i], bytes, cudaMemcpyDeviceToHost, streams[i]));
        
        std::cout << "  Stream " << i << ": Queued operations" << std::endl;
    }
    
    // Wait for all streams to complete
    std::cout << "Waiting for all streams to complete..." << std::endl;
    for (int i = 0; i < numStreams; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
        std::cout << "  Stream " << i << ": Completed" << std::endl;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Total time with " << numStreams << " streams: " << duration.count() << " ms" << std::endl;
    
    // Verify one result
    bool correct = true;
    if (numStreams > 0) {
        // Just check first element of first stream
        float expected = h_a[0][0] + h_b[0][0];
        for (int iter = 0; iter < 500; iter++) {
            expected = expected * 1.001f + sinf(expected * 0.01f);
            expected = sqrtf(fabsf(expected));
        }
        if (abs(h_c[0][0] - expected) > 0.1f) {
            correct = false;
        }
    }
    std::cout << "Results: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
    
    // Cleanup
    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFreeHost(h_a[i]);
        cudaFreeHost(h_b[i]);
        cudaFreeHost(h_c[i]);
        cudaFree(d_a[i]);
        cudaFree(d_b[i]);
        cudaFree(d_c[i]);
    }
}

void demonstrateCopyEngineLimit() {
    std::cout << "\n=== Demonstrating Copy Engine Limitation ===" << std::endl;
    std::cout << "A100 has 5 copy engines, so only 5 memory transfers can happen simultaneously" << std::endl;
    
    const int numStreams = 10;  // More streams than copy engines
    const int N = 1024 * 1024;
    const int bytes = N * sizeof(float);
    
    std::vector<cudaStream_t> streams(numStreams);
    std::vector<float*> h_data(numStreams), d_data(numStreams);
    std::vector<cudaEvent_t> startEvents(numStreams), endEvents(numStreams);
    
    // Create streams and events
    for (int i = 0; i < numStreams; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
        CHECK_CUDA(cudaEventCreate(&startEvents[i]));
        CHECK_CUDA(cudaEventCreate(&endEvents[i]));
        
        CHECK_CUDA(cudaMallocHost(&h_data[i], bytes));
        CHECK_CUDA(cudaMalloc(&d_data[i], bytes));
        
        // Initialize data
        for (int j = 0; j < N; j++) {
            h_data[i][j] = i + j * 0.001f;
        }
    }
    
    std::cout << "Launching " << numStreams << " concurrent memory transfers..." << std::endl;
    
    // Launch memory transfers with timing
    for (int i = 0; i < numStreams; i++) {
        CHECK_CUDA(cudaEventRecord(startEvents[i], streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_data[i], h_data[i], bytes, cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaEventRecord(endEvents[i], streams[i]));
    }
    
    // Wait and measure times
    for (int i = 0; i < numStreams; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
        
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, startEvents[i], endEvents[i]));
        std::cout << "  Stream " << i << " transfer time: " << milliseconds << " ms" << std::endl;
    }
    
    std::cout << "Notice: First 5 transfers should be fastest (using copy engines directly)" << std::endl;
    std::cout << "Later transfers may be queued and take longer" << std::endl;
    
    // Cleanup
    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(startEvents[i]);
        cudaEventDestroy(endEvents[i]);
        cudaFreeHost(h_data[i]);
        cudaFree(d_data[i]);
    }
}

void showA100Capabilities() {
    std::cout << "\n=== A100 GPU Capabilities ===" << std::endl;
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
    std::cout << "Streaming Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Copy Engines (asyncEngineCount): " << prop.asyncEngineCount << std::endl;
    std::cout << "Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    
    // Calculate theoretical values
    int maxConcurrentThreads = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
    std::cout << "Max concurrent threads: " << maxConcurrentThreads << std::endl;
    
    // Estimate practical stream limits
    std::cout << "\nPractical Stream Guidelines for A100:" << std::endl;
    std::cout << "- Memory-bound operations: Limited by " << prop.asyncEngineCount << " copy engines" << std::endl;
    std::cout << "- Compute-bound operations: Can use 50-100+ streams effectively" << std::endl;
    std::cout << "- Mixed workloads: 8-32 streams often optimal" << std::endl;
}

int main() {
    std::cout << "A100-Optimized CUDA Streams Demonstration" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Show A100 capabilities
    showA100Capabilities();
    
    // Test different numbers of streams
    std::vector<int> streamCounts = {1, 4, 8, 16, 32,64, 128};
    
    for (int numStreams : streamCounts) {
        runWithOptimalStreams(numStreams);
    }
    
    // Demonstrate copy engine limitation
    demonstrateCopyEngineLimit();
    
    std::cout << "\n=== Key Insights for A100 ===" << std::endl;
    std::cout << "1. A100 has 5 copy engines for memory transfers" << std::endl;
    std::cout << "2. But can run many more compute streams concurrently" << std::endl;
    std::cout << "3. Optimal stream count depends on memory vs compute ratio" << std::endl;
    std::cout << "4. More streams can help hide memory latency" << std::endl;
    std::cout << "5. Beyond ~32 streams, diminishing returns for most workloads" << std::endl;
    
    return 0;
}