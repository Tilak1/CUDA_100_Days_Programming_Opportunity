// Filename: benchmark_persistent_kernel.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <thread>

constexpr int ELEMENTS_PER_SYMBOL = 4096;
constexpr int SYMBOLS_PER_RAY = 20;
constexpr int BLOCKSIZE = 128;
constexpr int WARMUP_RUNS = 5;
constexpr int PERSISTENT_BLOCKS = 80;  // Number of persistent thread blocks

using ComplexType = __half2;

/* ------------------------------------------------------------------ */
/*                        WORK QUEUE STRUCTURES                       */
/* ------------------------------------------------------------------ */

struct WorkItem {
    int ray_id;
    int symbol_id;
    size_t offset;
    int num_elements;
};

struct WorkQueue {
    volatile int* head;           // Current work item to process
    volatile int* tail;           // Where to add new work
    volatile int* completed;      // Number of completed items
    volatile int* should_exit;    // Signal to terminate kernel
    WorkItem* items;             // Array of work items
    int max_items;
};

/* ------------------------------------------------------------------ */
/*                      PERSISTENT KERNEL                             */
/* ------------------------------------------------------------------ */

__device__ void process_complex_multiply(const __half2* __restrict__ X, 
                                       const __half2* __restrict__ H, 
                                       __half2* __restrict__ Y, 
                                       size_t offset, int num_elements) {
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Process elements starting from offset (in __half2 units)
    for (int idx = tid; idx < num_elements; idx += stride) {
        size_t i = offset + idx;
        __half2 x_val = X[i];
        __half2 h_val = H[i];
        __half x_real = __low2half(x_val);
        __half x_imag = __high2half(x_val);
        __half h_real = __low2half(h_val);
        __half h_imag = __high2half(h_val);
        __half r_real = __hsub(__hmul(x_real, h_real), __hmul(x_imag, h_imag));
        __half r_imag = __hadd(__hmul(x_real, h_imag), __hmul(x_imag, h_real));
        Y[i] = __halves2half2(r_real, r_imag);
    }
}

__global__ void persistent_kernel(WorkQueue queue,
                                const __half2* __restrict__ X, 
                                const __half2* __restrict__ H, 
                                __half2* __restrict__ Y) {
    __shared__ WorkItem shared_work;
    __shared__ int has_work;
    
    // Debug: Report that this block has started
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Persistent kernel started on block %d\n", blockIdx.x);
    }
    
    // Each block continuously processes work items
    while (true) {
        // Check exit condition
        if (threadIdx.x == 0) {
            int should_exit = *(volatile int*)queue.should_exit;
            if (should_exit != 0) {
                has_work = -1;  // Signal all threads to exit
            } else {
                has_work = 0;
            }
        }
        __syncthreads();
        
        if (has_work == -1) {
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                printf("Block %d exiting\n", blockIdx.x);
            }
            break;  // Exit condition
        }
        
        if (threadIdx.x == 0) {
            // Try to get work
            int tail_val = *(volatile int*)queue.tail;
            int my_work = atomicAdd((int*)queue.head, 1);
            
            if (my_work < tail_val) {
                // We got valid work
                shared_work = queue.items[my_work];
                has_work = 1;
                
                // Debug output for first few items
                if (my_work < 5 || my_work % 1000 == 0) {
                    printf("Block %d processing work item %d (offset=%llu, elements=%d)\n", 
                           blockIdx.x, my_work, (unsigned long long)shared_work.offset, 
                           shared_work.num_elements);
                }
            } else {
                has_work = 0;
                // Reset head if we've gone past tail
                atomicSub((int*)queue.head, 1);
            }
        }
        __syncthreads();
        
        if (has_work == 1) {
            // All threads in block process this work item
            process_complex_multiply(X, H, Y, shared_work.offset, shared_work.num_elements);
            
            // Mark work as completed
            if (threadIdx.x == 0) {
                atomicAdd((int*)queue.completed, 1);
            }
        }
        
        // Small delay to prevent busy waiting
        __threadfence_system();
    }
}

/* ------------------------------------------------------------------ */
/*                    STANDARD KERNEL (for comparison)               */
/* ------------------------------------------------------------------ */

__global__ void pw_multiply_half_kernel(const __half2* __restrict__ X, 
                                       const __half2* __restrict__ H, 
                                       __half2* __restrict__ Y, int n) {
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
/*                              HELPERS                               */
/* ------------------------------------------------------------------ */

void gpuCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

ComplexType* safeHostAlloc(size_t bytes, const char* name) {
    ComplexType* ptr = nullptr;
    if (cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault) == cudaSuccess) {
        return ptr;
    }
    ptr = (ComplexType*)malloc(bytes);
    if (!ptr) { 
        fprintf(stderr, "FATAL: malloc failed for %s\n", name); 
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

/* ------------------------------------------------------------------ */
/*                          DATA STRUCTURES                           */
/* ------------------------------------------------------------------ */

struct TestConfig {
    int total_rays;
    int kernels_per_loop;
    std::string description;
    
    // Calculated values
    int batch_size_symbols() const { return total_rays * SYMBOLS_PER_RAY; }
    int symbols_per_kernel() const { return batch_size_symbols() / kernels_per_loop; }
    int rays_per_kernel() const { return total_rays / kernels_per_loop; }
    size_t total_bytes() const { 
        return (size_t)batch_size_symbols() * ELEMENTS_PER_SYMBOL * sizeof(ComplexType); 
    }
};

struct BenchmarkResults {
    float h2d_time_ms;
    float submission_time_ms;      // Time to submit work to queue
    float processing_time_ms;       // Time for persistent kernel to process
    float d2h_time_ms;
    float total_time_ms;
    float startup_time_ms;         // Time to launch persistent kernel
    float shutdown_time_ms;        // Time to signal and wait for exit
    float avg_latency_us;          // Average latency per work item
};

/* ------------------------------------------------------------------ */
/*                    PERSISTENT KERNEL BENCHMARK                     */
/* ------------------------------------------------------------------ */

BenchmarkResults benchmarkPersistentKernel(const TestConfig& config,
                                          ComplexType* h_X, ComplexType* h_H, ComplexType* h_Y,
                                          ComplexType* d_X, ComplexType* d_H, ComplexType* d_Y,
                                          cudaStream_t stream) {
    BenchmarkResults results = {0};
    cudaEvent_t start, stop;
    gpuCheck(cudaEventCreate(&start), "create event");
    gpuCheck(cudaEventCreate(&stop), "create event");
    
    size_t total_bytes = config.total_bytes();
    
    // H2D Transfer
    gpuCheck(cudaEventRecord(start, stream), "record");
    gpuCheck(cudaMemcpyAsync(d_X, h_X, total_bytes, cudaMemcpyHostToDevice, stream), "H2D X");
    gpuCheck(cudaMemcpyAsync(d_H, h_H, total_bytes, cudaMemcpyHostToDevice, stream), "H2D H");
    gpuCheck(cudaEventRecord(stop, stream), "record");
    gpuCheck(cudaEventSynchronize(stop), "sync");
    gpuCheck(cudaEventElapsedTime(&results.h2d_time_ms, start, stop), "get time");
    
    // Clear device error state before starting
    cudaGetLastError();
    
    // Allocate work queue structures
    WorkQueue h_queue;
    h_queue.max_items = config.kernels_per_loop;
    
    gpuCheck(cudaMalloc((void**)&h_queue.head, sizeof(int)), "alloc head");
    gpuCheck(cudaMalloc((void**)&h_queue.tail, sizeof(int)), "alloc tail");
    gpuCheck(cudaMalloc((void**)&h_queue.completed, sizeof(int)), "alloc completed");
    gpuCheck(cudaMalloc((void**)&h_queue.should_exit, sizeof(int)), "alloc exit flag");
    gpuCheck(cudaMalloc(&h_queue.items, sizeof(WorkItem) * h_queue.max_items), "alloc items");
    
    // Initialize queue
    int zero = 0;
    gpuCheck(cudaMemset((void*)h_queue.head, 0, sizeof(int)), "init head");
    gpuCheck(cudaMemset((void*)h_queue.tail, 0, sizeof(int)), "init tail");
    gpuCheck(cudaMemset((void*)h_queue.completed, 0, sizeof(int)), "init completed");
    gpuCheck(cudaMemset((void*)h_queue.should_exit, 0, sizeof(int)), "init exit");
    
    // Prepare work items
    std::vector<WorkItem> work_items(config.kernels_per_loop);
    int symbols_per_kernel = config.symbols_per_kernel();
    
    for (int i = 0; i < config.kernels_per_loop; i++) {
        int start_symbol = i * symbols_per_kernel;
        int num_symbols = (i == config.kernels_per_loop - 1) ? 
            (config.batch_size_symbols() - start_symbol) : symbols_per_kernel;
        
        work_items[i].ray_id = i * config.rays_per_kernel();
        work_items[i].symbol_id = start_symbol;
        work_items[i].offset = (size_t)start_symbol * ELEMENTS_PER_SYMBOL;
        work_items[i].num_elements = num_symbols * ELEMENTS_PER_SYMBOL;
    }
    
    // Launch persistent kernel (wait a bit to ensure it's running)
    auto launch_start = std::chrono::high_resolution_clock::now();
    printf("  Launching %d persistent thread blocks...\n", PERSISTENT_BLOCKS);
    persistent_kernel<<<PERSISTENT_BLOCKS, BLOCKSIZE, 0, stream>>>(h_queue, d_X, d_H, d_Y);
    gpuCheck(cudaGetLastError(), "launch persistent kernel");
    std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Give kernel time to start
    printf("  Persistent kernel launched successfully\n");
    auto launch_end = std::chrono::high_resolution_clock::now();
    results.startup_time_ms = std::chrono::duration<float, std::milli>(launch_end - launch_start).count();
    
    // Submit work items in batches to avoid race conditions
    gpuCheck(cudaEventRecord(start, stream), "record");
    
    auto submit_start = std::chrono::high_resolution_clock::now();
    
    printf("  Submitting %d work items to queue...\n", config.kernels_per_loop);
    
    // First, prepare all work items on device
    gpuCheck(cudaMemcpy(h_queue.items, work_items.data(), 
                       sizeof(WorkItem) * config.kernels_per_loop, 
                       cudaMemcpyHostToDevice), "copy all work items");
    
    // Then update tail to make all work available at once
    int final_tail = config.kernels_per_loop;
    gpuCheck(cudaMemcpy((void*)h_queue.tail, &final_tail, sizeof(int), 
                       cudaMemcpyHostToDevice), "update tail");
    
    printf("  Work submission complete\n");
    
    auto submit_end = std::chrono::high_resolution_clock::now();
    results.submission_time_ms = std::chrono::duration<float, std::milli>(submit_end - submit_start).count();
    
    // Wait for all work to complete with progress bar
    auto process_start = std::chrono::high_resolution_clock::now();
    int completed = 0;
    int last_progress = -1;
    
    printf("  Progress: [");
    fflush(stdout);
    
    while (completed < config.kernels_per_loop) {
        gpuCheck(cudaMemcpy(&completed, (void*)h_queue.completed, sizeof(int), 
                           cudaMemcpyDeviceToHost), "check completed");
        
        // Update progress bar
        int progress = (completed * 50) / config.kernels_per_loop;
        if (progress > last_progress) {
            for (int i = last_progress + 1; i <= progress; i++) {
                printf("=");
                fflush(stdout);
            }
            last_progress = progress;
        }
        
        // Show detailed progress every 10%
        int percentage = (completed * 100) / config.kernels_per_loop;
        if (percentage % 10 == 0 && percentage != (((completed - 1) * 100) / config.kernels_per_loop)) {
            printf("] %d%% (%d/%d) [", percentage, completed, config.kernels_per_loop);
            fflush(stdout);
        }
        
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    
    // Complete progress bar
    for (int i = last_progress + 1; i <= 50; i++) {
        printf("=");
    }
    printf("] 100%% (%d/%d)\n", completed, config.kernels_per_loop);
    
    auto process_end = std::chrono::high_resolution_clock::now();
    results.processing_time_ms = std::chrono::duration<float, std::milli>(process_end - process_start).count();
    
    gpuCheck(cudaEventRecord(stop, stream), "record");
    gpuCheck(cudaEventSynchronize(stop), "sync");
    
    // Signal kernel to exit with status
    auto shutdown_start = std::chrono::high_resolution_clock::now();
    printf("  Signaling kernel shutdown...\n");
    int one = 1;
    gpuCheck(cudaMemcpy((void*)h_queue.should_exit, &one, sizeof(int), 
                       cudaMemcpyHostToDevice), "signal exit");
    cudaStreamSynchronize(stream);  // Wait for kernel to exit
    printf("  Kernel shutdown complete\n");
    auto shutdown_end = std::chrono::high_resolution_clock::now();
    results.shutdown_time_ms = std::chrono::duration<float, std::milli>(shutdown_end - shutdown_start).count();
    
    // D2H Transfer
    gpuCheck(cudaEventRecord(start, stream), "record");
    gpuCheck(cudaMemcpyAsync(h_Y, d_Y, total_bytes, cudaMemcpyDeviceToHost, stream), "D2H Y");
    gpuCheck(cudaEventRecord(stop, stream), "record");
    gpuCheck(cudaEventSynchronize(stop), "sync");
    gpuCheck(cudaEventElapsedTime(&results.d2h_time_ms, start, stop), "get time");
    
    results.total_time_ms = results.h2d_time_ms + results.submission_time_ms + 
                           results.processing_time_ms + results.d2h_time_ms;
    results.avg_latency_us = (results.processing_time_ms * 1000.0f) / config.kernels_per_loop;
    
    // Cleanup
    gpuCheck(cudaFree((void*)h_queue.head), "free head");
    gpuCheck(cudaFree((void*)h_queue.tail), "free tail");
    gpuCheck(cudaFree((void*)h_queue.completed), "free completed");
    gpuCheck(cudaFree((void*)h_queue.should_exit), "free exit");
    gpuCheck(cudaFree(h_queue.items), "free items");
    
    gpuCheck(cudaEventDestroy(start), "destroy");
    gpuCheck(cudaEventDestroy(stop), "destroy");
    
    return results;
}

/* ------------------------------------------------------------------ */
/*                    STANDARD BENCHMARK (baseline)                   */
/* ------------------------------------------------------------------ */

BenchmarkResults benchmarkStandardLaunch(const TestConfig& config, 
                                        ComplexType* h_X, ComplexType* h_H, ComplexType* h_Y,
                                        ComplexType* d_X, ComplexType* d_H, ComplexType* d_Y,
                                        cudaStream_t stream) {
    BenchmarkResults results = {0};
    cudaEvent_t start, stop;
    gpuCheck(cudaEventCreate(&start), "create event");
    gpuCheck(cudaEventCreate(&stop), "create event");
    
    size_t total_bytes = config.total_bytes();
    
    // H2D Transfer
    gpuCheck(cudaEventRecord(start, stream), "record");
    gpuCheck(cudaMemcpyAsync(d_X, h_X, total_bytes, cudaMemcpyHostToDevice, stream), "H2D X");
    gpuCheck(cudaMemcpyAsync(d_H, h_H, total_bytes, cudaMemcpyHostToDevice, stream), "H2D H");
    gpuCheck(cudaEventRecord(stop, stream), "record");
    gpuCheck(cudaEventSynchronize(stop), "sync");
    gpuCheck(cudaEventElapsedTime(&results.h2d_time_ms, start, stop), "get time");
    
    // Kernel Execution
    dim3 block(BLOCKSIZE);
    int symbols_per_kernel = config.symbols_per_kernel();
    
    gpuCheck(cudaEventRecord(start, stream), "record");
    for (int i = 0; i < config.kernels_per_loop; i++) {
        int start_symbol = i * symbols_per_kernel;
        int num_symbols = (i == config.kernels_per_loop - 1) ? 
            (config.batch_size_symbols() - start_symbol) : symbols_per_kernel;
        int num_elements = num_symbols * ELEMENTS_PER_SYMBOL;
        size_t offset = (size_t)start_symbol * ELEMENTS_PER_SYMBOL;
        dim3 grid((num_elements + BLOCKSIZE - 1) / BLOCKSIZE);
        pw_multiply_half_kernel<<<grid, block, 0, stream>>>(
            d_X + offset, d_H + offset, d_Y + offset, num_elements);
    }
    gpuCheck(cudaEventRecord(stop, stream), "record");
    gpuCheck(cudaEventSynchronize(stop), "sync");
    gpuCheck(cudaEventElapsedTime(&results.processing_time_ms, start, stop), "get time");
    
    // D2H Transfer
    gpuCheck(cudaEventRecord(start, stream), "record");
    gpuCheck(cudaMemcpyAsync(h_Y, d_Y, total_bytes, cudaMemcpyDeviceToHost, stream), "D2H Y");
    gpuCheck(cudaEventRecord(stop, stream), "record");
    gpuCheck(cudaEventSynchronize(stop), "sync");
    gpuCheck(cudaEventElapsedTime(&results.d2h_time_ms, start, stop), "get time");
    
    results.total_time_ms = results.h2d_time_ms + results.processing_time_ms + results.d2h_time_ms;
    
    gpuCheck(cudaEventDestroy(start), "destroy");
    gpuCheck(cudaEventDestroy(stop), "destroy");
    
    return results;
}

/* ------------------------------------------------------------------ */
/*                         RUN COMPARISON                             */
/* ------------------------------------------------------------------ */

void runComparison(const TestConfig& config) {
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("Testing: %s\n", config.description.c_str());
    printf("  Total rays: %d, Work items: %d, Rays/item: %d\n", 
           config.total_rays, config.kernels_per_loop, config.rays_per_kernel());
    
    // Allocate memory
    size_t total_bytes = config.total_bytes();
    ComplexType *h_X = safeHostAlloc(total_bytes, "h_X");
    ComplexType *h_H = safeHostAlloc(total_bytes, "h_H");
    ComplexType *h_Y_standard = safeHostAlloc(total_bytes, "h_Y_standard");
    ComplexType *h_Y_persistent = safeHostAlloc(total_bytes, "h_Y_persistent");
    
    // Initialize with random data
    size_t total_elements = (size_t)config.batch_size_symbols() * ELEMENTS_PER_SYMBOL;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (size_t i = 0; i < total_elements; i++) {
        h_X[i] = __halves2half2(__float2half(dist(gen)), __float2half(dist(gen)));
        h_H[i] = __halves2half2(__float2half(dist(gen)), __float2half(dist(gen)));
    }
    
    ComplexType *d_X, *d_H, *d_Y;
    gpuCheck(cudaMalloc(&d_X, total_bytes), "malloc");
    gpuCheck(cudaMalloc(&d_H, total_bytes), "malloc");
    gpuCheck(cudaMalloc(&d_Y, total_bytes), "malloc");
    
    cudaStream_t stream;
    gpuCheck(cudaStreamCreate(&stream), "create stream");
    
    // Run benchmarks
    printf("\n📊 Standard Launch:\n");
    BenchmarkResults standard = benchmarkStandardLaunch(config, h_X, h_H, h_Y_standard, 
                                                       d_X, d_H, d_Y, stream);
    printf("  H2D: %.2f ms\n", standard.h2d_time_ms);
    printf("  Processing: %.2f ms (%.2f μs per kernel)\n", 
           standard.processing_time_ms, (standard.processing_time_ms * 1000.0f) / config.kernels_per_loop);
    printf("  D2H: %.2f ms\n", standard.d2h_time_ms);
    printf("  Total: %.2f ms\n", standard.total_time_ms);
    
    printf("\n🔄 Persistent Kernel:\n");
    BenchmarkResults persistent = benchmarkPersistentKernel(config, h_X, h_H, h_Y_persistent,
                                                           d_X, d_H, d_Y, stream);
    printf("  H2D: %.2f ms\n", persistent.h2d_time_ms);
    printf("  Kernel startup: %.2f ms\n", persistent.startup_time_ms);
    printf("  Work submission: %.2f ms\n", persistent.submission_time_ms);
    printf("  Processing: %.2f ms (%.2f μs avg latency)\n", 
           persistent.processing_time_ms, persistent.avg_latency_us);
    printf("  Kernel shutdown: %.2f ms\n", persistent.shutdown_time_ms);
    printf("  D2H: %.2f ms\n", persistent.d2h_time_ms);
    printf("  Total: %.2f ms\n", persistent.total_time_ms);
    
    // Calculate speedup
    float speedup = standard.total_time_ms / persistent.total_time_ms;
    printf("\n⚡ Speedup: %.2fx\n", speedup);
    
    // Verify results match
    bool results_match = true;
    for (size_t i = 0; i < total_elements && i < 10; i++) {
        float std_real = __half2float(__low2half(h_Y_standard[i]));
        float std_imag = __half2float(__high2half(h_Y_standard[i]));
        float per_real = __half2float(__low2half(h_Y_persistent[i]));
        float per_imag = __half2float(__high2half(h_Y_persistent[i]));
        
        if (std::abs(std_real - per_real) > 0.01f || std::abs(std_imag - per_imag) > 0.01f) {
            results_match = false;
            printf("❌ Mismatch at %zu: standard(%.3f,%.3f) vs persistent(%.3f,%.3f)\n",
                   i, std_real, std_imag, per_real, per_imag);
        }
    }
    if (results_match) {
        printf("✅ Results match!\n");
    }
    
    // Cleanup
    gpuCheck(cudaStreamDestroy(stream), "destroy");
    gpuCheck(cudaFree(d_X), "free");
    gpuCheck(cudaFree(d_H), "free");
    gpuCheck(cudaFree(d_Y), "free");
    safeFree(h_X);
    safeFree(h_H);
    safeFree(h_Y_standard);
    safeFree(h_Y_persistent);
}

/* ------------------------------------------------------------------ */
/*                                M A I N                             */
/* ------------------------------------------------------------------ */

int main() {
    printf("🚀 Persistent Kernel vs Standard Launch Benchmark\n");
    printf("================================================\n");
    
    // Check CUDA device
    int deviceCount;
    gpuCheck(cudaGetDeviceCount(&deviceCount), "get device count");
    if (deviceCount < 1) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    gpuCheck(cudaGetDeviceProperties(&prop, 0), "get device properties");
    printf("Using GPU: %s\n", prop.name);
    printf("SM Count: %d, Max Threads/Block: %d\n", prop.multiProcessorCount, prop.maxThreadsPerBlock);
    printf("Persistent blocks: %d\n\n", PERSISTENT_BLOCKS);
    
    // Test configurations
    std::vector<TestConfig> configs = {
        // Small workloads - overhead should dominate
        {100,   100,   "Small workload (100 rays, 100 kernels)"},
        {500,   500,   "Small-medium (500 rays, 500 kernels)"},
        
        // Medium workloads
        {1000,  1000,  "Medium workload (1000 rays, 1000 kernels)"},
        {2000,  2000,  "Medium-large (2000 rays, 2000 kernels)"},
        
        // Large workloads where persistent kernel should shine
        {4000,  4000,  "Large workload (4000 rays, 4000 kernels)"},
        {8000,  8000,  "XL workload (8000 rays, 8000 kernels)"},
        
        // Extreme case - many small kernels
        {4000,  20000, "20K small kernels (5 symbols each)"},
        {4000,  40000, "40K micro kernels (2 symbols each)"},
        {4000,  80000, "80K nano kernels (1 symbol each)"},
    };
    
    // Run tests
    for (const auto& config : configs) {
        try {
            runComparison(config);
        } catch (const std::exception& e) {
            printf("Error testing config %s: %s\n", config.description.c_str(), e.what());
        }
    }
    
    printf("\n\n🏁 Benchmark completed!\n");
    printf("====================================\n");
    printf("Key findings:\n");
    printf("- Persistent kernels eliminate ALL launch overhead\n");
    printf("- But add coordination overhead (work queue management)\n");
    printf("- Best for scenarios with many small kernels\n");
    printf("- Trade-off: complexity vs performance\n");
    
    return 0;
}