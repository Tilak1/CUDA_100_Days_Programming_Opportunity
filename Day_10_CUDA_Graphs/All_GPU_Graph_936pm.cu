// Filename: benchmark_graphs_everywhere_corrected.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <atomic>

constexpr int ELEMENTS_PER_SYMBOL = 4096;
constexpr int SYMBOLS_PER_RAY = 20;
constexpr int TOTAL_RAYS = 1000;
constexpr int BLOCKSIZE = 128;
constexpr int KERNELS_PER_GPU = 4;

constexpr int BATCH_SIZE_SYMBOLS = TOTAL_RAYS * SYMBOLS_PER_RAY;

using ComplexType = __half2;

/* ------------------------------------------------------------------ */
/*                            K E R N E L                             */
/* ------------------------------------------------------------------ */

__global__ void pw_multiply_half_kernel(const __half2* __restrict__ X, 
                                       const __half2* __restrict__ H, 
                                       __half2* __restrict__ Y, 
                                       int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        __half2 x_val = X[i], h_val = H[i];
        __half x_real = __low2half(x_val), x_imag = __high2half(x_val);
        __half h_real = __low2half(h_val), h_imag = __high2half(h_val);
        __half r_real = __hsub(__hmul(x_real, h_real), __hmul(x_imag, h_imag));
        __half r_imag = __hadd(__hmul(x_real, h_imag), __hmul(x_imag, h_real));
        Y[i] = __halves2half2(r_real, r_imag);
    }
}

/* ------------------------------------------------------------------ */
/*                              H E L P E R S                         */
/* ------------------------------------------------------------------ */

void gpuCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Thread-safe error checking for multi-threaded sections
bool gpuCheckThreadSafe(cudaError_t err, const char* msg, int device) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error on device %d - %s : %s\n", 
                device, msg, cudaGetErrorString(err));
        return false;
    }
    return true;
}

std::vector<__half2> randomComplexHalfVector(int n, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<__half2> v(n);
    for (auto& c : v) {
        c = __halves2half2(__float2half(dist(gen)), __float2half(dist(gen)));
    }
    return v;
}

ComplexType* safeHostAlloc(size_t bytes, const char* name) {
    ComplexType* ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (err == cudaSuccess && ptr != nullptr) {
        printf("  %s: cudaHostAlloc success (%.2f GB)\n", name, bytes / (1e9));
        return ptr;
    }
    
    // Fallback to regular malloc
    printf("  %s: cudaHostAlloc failed, using malloc (%.2f GB)\n", name, bytes / (1e9));
    ptr = (ComplexType*)malloc(bytes);
    if (!ptr) {
        fprintf(stderr, "FATAL: Failed to allocate %s (%.2f GB)\n", name, bytes / (1e9));
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void safeFree(ComplexType* ptr) {
    if (ptr) {
        if (cudaFreeHost(ptr) != cudaSuccess) {
            free(ptr);
        }
    }
}

struct TimingResults {
    float total_time;
};

/* ------------------------------------------------------------------ */
/*                                M A I N                             */
/* ------------------------------------------------------------------ */

int main() {
    int deviceCount;
    gpuCheck(cudaGetDeviceCount(&deviceCount), "get device count");
    if (deviceCount < 1) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    printf("=== Graph vs. Graph Performance Analysis ===\n");
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    long long total_ops = (long long)BATCH_SIZE_SYMBOLS * ELEMENTS_PER_SYMBOL;
    printf("Workload: %.1f million complex multiplies\n", total_ops / 1e6);
    printf("Total symbols: %d, Elements per symbol: %d\n\n", BATCH_SIZE_SYMBOLS, ELEMENTS_PER_SYMBOL);

    // Allocate host memory
    size_t total_bytes = (size_t)BATCH_SIZE_SYMBOLS * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
    ComplexType *h_X = safeHostAlloc(total_bytes, "h_X");
    ComplexType *h_H = safeHostAlloc(total_bytes, "h_H");
    ComplexType *h_Y = safeHostAlloc(total_bytes, "h_Y");

    // Initialize data
    printf("Initializing data...\n");
    for (int i = 0; i < BATCH_SIZE_SYMBOLS; i++) {
        auto vec_X = randomComplexHalfVector(ELEMENTS_PER_SYMBOL, i * 2);
        auto vec_H = randomComplexHalfVector(ELEMENTS_PER_SYMBOL, i * 2 + 1);
        size_t offset = (size_t)i * ELEMENTS_PER_SYMBOL;
        memcpy(&h_X[offset], vec_X.data(), ELEMENTS_PER_SYMBOL * sizeof(ComplexType));
        memcpy(&h_H[offset], vec_H.data(), ELEMENTS_PER_SYMBOL * sizeof(ComplexType));
    }

    // Single-GPU Graph Benchmark
    TimingResults singleGpuResults = {0};
    {
        printf("\n--- Single-GPU Graph Benchmark ---\n");
        gpuCheck(cudaSetDevice(0), "set dev 0");
        
        ComplexType *d_X, *d_H, *d_Y;
        gpuCheck(cudaMalloc(&d_X, total_bytes), "malloc X");
        gpuCheck(cudaMalloc(&d_H, total_bytes), "malloc H");
        gpuCheck(cudaMalloc(&d_Y, total_bytes), "malloc Y");

        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        gpuCheck(cudaStreamCreate(&stream), "create stream");
        
        // Begin graph capture
        gpuCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
        
        // Memory transfers
        gpuCheck(cudaMemcpyAsync(d_X, h_X, total_bytes, cudaMemcpyHostToDevice, stream), "H2D X");
        gpuCheck(cudaMemcpyAsync(d_H, h_H, total_bytes, cudaMemcpyHostToDevice, stream), "H2D H");
        
        // Launch kernels
        dim3 block(BLOCKSIZE);
        int symbols_per_kernel = BATCH_SIZE_SYMBOLS / KERNELS_PER_GPU;
        
        for (int i = 0; i < KERNELS_PER_GPU; i++) {
            int start_symbol = i * symbols_per_kernel;
            int num_symbols = (i == KERNELS_PER_GPU - 1) ? 
                (BATCH_SIZE_SYMBOLS - start_symbol) : symbols_per_kernel;
            
            if (num_symbols <= 0) continue;
            
            int num_elements = num_symbols * ELEMENTS_PER_SYMBOL;
            size_t offset = (size_t)start_symbol * ELEMENTS_PER_SYMBOL;
            dim3 grid((num_elements + BLOCKSIZE - 1) / BLOCKSIZE);
            
            // FIXED: Use direct pointer arithmetic
            pw_multiply_half_kernel<<<grid, block, 0, stream>>>(
                d_X + offset,
                d_H + offset,
                d_Y + offset,
                num_elements);
        }
        
        // Memory transfer back
        gpuCheck(cudaMemcpyAsync(h_Y, d_Y, total_bytes, cudaMemcpyDeviceToHost, stream), "D2H Y");
        
        // End capture and instantiate
        gpuCheck(cudaStreamEndCapture(stream, &graph), "end capture");
        gpuCheck(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0), "instantiate");

        // Time the execution
        cudaEvent_t start, stop;
        gpuCheck(cudaEventCreate(&start), "create start");
        gpuCheck(cudaEventCreate(&stop), "create stop");

        gpuCheck(cudaEventRecord(start, stream), "record start");
        gpuCheck(cudaGraphLaunch(graphExec, stream), "launch graph");
        gpuCheck(cudaEventRecord(stop, stream), "record stop");
        gpuCheck(cudaEventSynchronize(stop), "sync");
        
        gpuCheck(cudaEventElapsedTime(&singleGpuResults.total_time, start, stop), "get time");
        
        printf("Single GPU execution time: %.2f ms\n", singleGpuResults.total_time);
        
        // Cleanup
        gpuCheck(cudaFree(d_X), "free d_X");
        gpuCheck(cudaFree(d_H), "free d_H");
        gpuCheck(cudaFree(d_Y), "free d_Y");
        gpuCheck(cudaGraphExecDestroy(graphExec), "destroy exec");
        gpuCheck(cudaGraphDestroy(graph), "destroy graph");
        gpuCheck(cudaStreamDestroy(stream), "destroy stream");
        gpuCheck(cudaEventDestroy(start), "destroy start");
        gpuCheck(cudaEventDestroy(stop), "destroy stop");
    }

    // Multi-GPU Graph Benchmark
    TimingResults multiGpuResults = {0};
    {
        printf("\n--- Multi-GPU Graph Benchmark ---\n");
        std::vector<cudaGraph_t> graphs(deviceCount);
        std::vector<cudaGraphExec_t> graph_execs(deviceCount);
        std::vector<cudaStream_t> streams(deviceCount);
        std::vector<ComplexType*> d_X(deviceCount), d_H(deviceCount), d_Y(deviceCount);

        // FIXED: Track symbols, not elements
        size_t symbols_processed = 0;
        int base_symbols = BATCH_SIZE_SYMBOLS / deviceCount;
        int remainder_symbols = BATCH_SIZE_SYMBOLS % deviceCount;

        // Create graphs for each GPU
        for (int dev = 0; dev < deviceCount; ++dev) {
            gpuCheck(cudaSetDevice(dev), "set dev");
            gpuCheck(cudaStreamCreate(&streams[dev]), "create stream");

            int gpu_symbols = base_symbols + (dev < remainder_symbols ? 1 : 0);
            size_t gpu_elements = (size_t)gpu_symbols * ELEMENTS_PER_SYMBOL;
            size_t gpu_bytes = gpu_elements * sizeof(ComplexType);
            size_t offset_elements = symbols_processed * ELEMENTS_PER_SYMBOL;
            
            printf("GPU %d: Processing %d symbols (%zu elements)\n", dev, gpu_symbols, gpu_elements);
            
            gpuCheck(cudaMalloc(&d_X[dev], gpu_bytes), "malloc X");
            gpuCheck(cudaMalloc(&d_H[dev], gpu_bytes), "malloc H");
            gpuCheck(cudaMalloc(&d_Y[dev], gpu_bytes), "malloc Y");

            // FIXED: Use ThreadLocal for multi-GPU capture
            gpuCheck(cudaStreamBeginCapture(streams[dev], cudaStreamCaptureModeThreadLocal), "begin capture");
            
            // FIXED: Use correct offset
            gpuCheck(cudaMemcpyAsync(d_X[dev], h_X + offset_elements, gpu_bytes, 
                cudaMemcpyHostToDevice, streams[dev]), "H2D X");
            gpuCheck(cudaMemcpyAsync(d_H[dev], h_H + offset_elements, gpu_bytes, 
                cudaMemcpyHostToDevice, streams[dev]), "H2D H");

            // Launch kernels
            dim3 block(BLOCKSIZE);
            int symbols_per_kernel = gpu_symbols / KERNELS_PER_GPU;
            
            for (int i = 0; i < KERNELS_PER_GPU; i++) {
                int start_symbol_local = i * symbols_per_kernel;
                int num_symbols = (i == KERNELS_PER_GPU - 1) ? 
                    (gpu_symbols - start_symbol_local) : symbols_per_kernel;
                
                if (num_symbols <= 0) continue;
                
                int num_elements = num_symbols * ELEMENTS_PER_SYMBOL;
                size_t offset = (size_t)start_symbol_local * ELEMENTS_PER_SYMBOL;
                dim3 grid((num_elements + BLOCKSIZE - 1) / BLOCKSIZE);
                
                // FIXED: Use direct pointer arithmetic
                pw_multiply_half_kernel<<<grid, block, 0, streams[dev]>>>(
                    d_X[dev] + offset,
                    d_H[dev] + offset,
                    d_Y[dev] + offset,
                    num_elements);
            }
            
            // FIXED: Use correct offset
            gpuCheck(cudaMemcpyAsync(h_Y + offset_elements, d_Y[dev], gpu_bytes, 
                cudaMemcpyDeviceToHost, streams[dev]), "D2H Y");
            
            gpuCheck(cudaStreamEndCapture(streams[dev], &graphs[dev]), "end capture");
            gpuCheck(cudaGraphInstantiate(&graph_execs[dev], graphs[dev], NULL, NULL, 0), "instantiate");
            
            // FIXED: Increment by symbols, not elements
            symbols_processed += gpu_symbols;
        }
        
        // Time the execution
        cudaEvent_t start, stop;
        gpuCheck(cudaEventCreate(&start), "create start");
        gpuCheck(cudaEventCreate(&stop), "create stop");
        gpuCheck(cudaEventRecord(start, 0), "record start");

        // FIXED: Better error handling for threaded launch
        std::vector<std::thread> launch_threads;
        std::vector<cudaError_t> thread_errors(deviceCount, cudaSuccess);
        std::atomic<bool> has_error(false);
        
        for (int dev = 0; dev < deviceCount; ++dev) {
            launch_threads.emplace_back([=, &graph_execs, &streams, &thread_errors, &has_error] {
                cudaError_t err = cudaSetDevice(dev);
                if (err == cudaSuccess) {
                    err = cudaGraphLaunch(graph_execs[dev], streams[dev]);
                }
                thread_errors[dev] = err;
                if (err != cudaSuccess) {
                    has_error.store(true);
                }
            });
        }
        
        // Wait for all threads to complete
        for (auto& t : launch_threads) {
            t.join();
        }
        
        // Check for errors
        if (has_error.load()) {
            for (int dev = 0; dev < deviceCount; ++dev) {
                if (thread_errors[dev] != cudaSuccess) {
                    fprintf(stderr, "Error on device %d: %s\n", 
                        dev, cudaGetErrorString(thread_errors[dev]));
                }
            }
            // Clean up and exit
            for (int dev = 0; dev < deviceCount; ++dev) {
                cudaSetDevice(dev);
                if (d_X[dev]) cudaFree(d_X[dev]);
                if (d_H[dev]) cudaFree(d_H[dev]);
                if (d_Y[dev]) cudaFree(d_Y[dev]);
                if (graph_execs[dev]) cudaGraphExecDestroy(graph_execs[dev]);
                if (graphs[dev]) cudaGraphDestroy(graphs[dev]);
                if (streams[dev]) cudaStreamDestroy(streams[dev]);
            }
            safeFree(h_X);
            safeFree(h_H);
            safeFree(h_Y);
            exit(EXIT_FAILURE);
        }
        
        // Synchronize all streams
        for (int dev = 0; dev < deviceCount; ++dev) {
            gpuCheck(cudaSetDevice(dev), "set dev");
            gpuCheck(cudaStreamSynchronize(streams[dev]), "sync stream");
        }
        
        gpuCheck(cudaEventRecord(stop, 0), "record stop");
        gpuCheck(cudaEventSynchronize(stop), "sync");
        gpuCheck(cudaEventElapsedTime(&multiGpuResults.total_time, start, stop), "get time");
        
        printf("Multi-GPU execution time: %.2f ms\n", multiGpuResults.total_time);

        // Cleanup
        for (int dev = 0; dev < deviceCount; ++dev) {
            gpuCheck(cudaSetDevice(dev), "set dev");
            gpuCheck(cudaFree(d_X[dev]), "free d_X");
            gpuCheck(cudaFree(d_H[dev]), "free d_H");
            gpuCheck(cudaFree(d_Y[dev]), "free d_Y");
            gpuCheck(cudaGraphExecDestroy(graph_execs[dev]), "destroy exec");
            gpuCheck(cudaGraphDestroy(graphs[dev]), "destroy graph");
            gpuCheck(cudaStreamDestroy(streams[dev]), "destroy stream");
        }
        gpuCheck(cudaEventDestroy(start), "destroy start");
        gpuCheck(cudaEventDestroy(stop), "destroy stop");
    }

    // Final Comparison
    printf("\n=== FINAL PERFORMANCE COMPARISON (Graph vs. Graph) ===\n");
    printf("                          Single GPU    Multi-GPU (%d)\n", deviceCount);
    printf("                          -----------    --------------\n");
    printf("Total Time (ms):        %8.2f       %8.2f\n", 
        singleGpuResults.total_time, multiGpuResults.total_time);
    
    float speedup = singleGpuResults.total_time / multiGpuResults.total_time;
    printf("\nSpeedup: %.2fx\n", speedup);
    printf("Parallel Efficiency: %.1f%%\n", (speedup / deviceCount) * 100.0);
    
    // Verify results (optional - compare a few elements)
    printf("\nVerifying results (first 5 elements):\n");
    bool results_match = true;
    for (int i = 0; i < 5 && i < BATCH_SIZE_SYMBOLS * ELEMENTS_PER_SYMBOL; i++) {
        __half2 val = h_Y[i];
        printf("Y[%d] = (%.3f, %.3f)\n", i, 
            __half2float(__low2half(val)), __half2float(__high2half(val)));
    }
    
    // Cleanup
    safeFree(h_X);
    safeFree(h_H);
    safeFree(h_Y);
    
    printf("\nBenchmark completed successfully!\n");
    return 0;
}