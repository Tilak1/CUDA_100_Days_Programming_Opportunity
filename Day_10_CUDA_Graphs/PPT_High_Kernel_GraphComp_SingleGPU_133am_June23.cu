// Filename: benchmark_detailed_breakdown_improved.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

constexpr int ELEMENTS_PER_SYMBOL = 4096;
constexpr int SYMBOLS_PER_RAY = 20;
constexpr int TOTAL_RAYS = 4000;
constexpr int BLOCKSIZE = 128;
constexpr int KERNELS_PER_LAUNCH_LOOP = 800000;
constexpr int WARMUP_RUNS = 10;
constexpr int TIMING_RUNS = 100;

constexpr int BATCH_SIZE_SYMBOLS = TOTAL_RAYS * SYMBOLS_PER_RAY;

using ComplexType = __half2;

/* ------------------------------------------------------------------ */
/*                            K E R N E L                             */
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
/*                              H E L P E R S                         */
/* ------------------------------------------------------------------ */

void gpuCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
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
    if (cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault) == cudaSuccess) {
        printf("  %s: cudaHostAlloc success (%.2f GB)\n", name, bytes / 1e9);
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

struct TimingResults {
    float h2d_time;
    float kernel_launch_time;     // CPU overhead for launching
    float kernel_exec_time;       // GPU execution time
    float d2h_time;
    float total_time;
    
    // Additional metrics
    float graph_creation_time;
    float per_kernel_launch_us;   // Microseconds per kernel launch
};

/* ------------------------------------------------------------------ */
/*                          TIMING FUNCTIONS                          */
/* ------------------------------------------------------------------ */

// Measure pure CPU launch overhead using high-resolution CPU timer
float measureCpuLaunchOverhead(cudaStream_t stream, int num_launches) {
    // Create a minimal kernel
    dim3 grid(1), block(1);
    
    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        pw_multiply_half_kernel<<<grid, block, 0, stream>>>(nullptr, nullptr, nullptr, 0);
    }
    cudaStreamSynchronize(stream);
    
    // Measure
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_launches; i++) {
        pw_multiply_half_kernel<<<grid, block, 0, stream>>>(nullptr, nullptr, nullptr, 0);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<float, std::milli> duration = end - start;
    return duration.count() / num_launches; // ms per launch
}

/* ------------------------------------------------------------------ */
/*                       BENCHMARK FUNCTIONS                          */
/* ------------------------------------------------------------------ */

TimingResults benchmarkStandardLaunch(ComplexType* h_X, ComplexType* h_H, ComplexType* h_Y,
                                     ComplexType* d_X, ComplexType* d_H, ComplexType* d_Y,
                                     size_t total_bytes, cudaStream_t stream) {
    TimingResults results = {0};
    cudaEvent_t start, stop;
    gpuCheck(cudaEventCreate(&start), "create event");
    gpuCheck(cudaEventCreate(&stop), "create event");
    
    printf("\n--- Benchmarking Standard Launch ---\n");
    
    // H2D Transfer
    gpuCheck(cudaEventRecord(start, stream), "record");
    gpuCheck(cudaMemcpyAsync(d_X, h_X, total_bytes, cudaMemcpyHostToDevice, stream), "H2D X");
    gpuCheck(cudaMemcpyAsync(d_H, h_H, total_bytes, cudaMemcpyHostToDevice, stream), "H2D H");
    gpuCheck(cudaEventRecord(stop, stream), "record");
    gpuCheck(cudaEventSynchronize(stop), "sync");
    gpuCheck(cudaEventElapsedTime(&results.h2d_time, start, stop), "get time");
    
    // Measure CPU launch overhead separately
    results.kernel_launch_time = measureCpuLaunchOverhead(stream, KERNELS_PER_LAUNCH_LOOP);
    results.per_kernel_launch_us = results.kernel_launch_time * 1000.0f; // Convert to microseconds
    
    // Kernel Execution (GPU time only)
    dim3 block(BLOCKSIZE);
    int symbols_per_kernel = BATCH_SIZE_SYMBOLS / KERNELS_PER_LAUNCH_LOOP;
    
    // Warmup
    for (int warm = 0; warm < WARMUP_RUNS; warm++) {
        for (int i = 0; i < KERNELS_PER_LAUNCH_LOOP; i++) {
            int start_symbol = i * symbols_per_kernel;
            int num_symbols = (i == KERNELS_PER_LAUNCH_LOOP - 1) ? 
                (BATCH_SIZE_SYMBOLS - start_symbol) : symbols_per_kernel;
            int num_elements = num_symbols * ELEMENTS_PER_SYMBOL;
            size_t offset = (size_t)start_symbol * ELEMENTS_PER_SYMBOL;
            dim3 grid((num_elements + BLOCKSIZE - 1) / BLOCKSIZE);
            pw_multiply_half_kernel<<<grid, block, 0, stream>>>(
                d_X + offset, d_H + offset, d_Y + offset, num_elements);
        }
    }
    cudaStreamSynchronize(stream);
    
    // Time actual kernel execution
    gpuCheck(cudaEventRecord(start, stream), "record");
    for (int i = 0; i < KERNELS_PER_LAUNCH_LOOP; i++) {
        int start_symbol = i * symbols_per_kernel;
        int num_symbols = (i == KERNELS_PER_LAUNCH_LOOP - 1) ? 
            (BATCH_SIZE_SYMBOLS - start_symbol) : symbols_per_kernel;
        int num_elements = num_symbols * ELEMENTS_PER_SYMBOL;
        size_t offset = (size_t)start_symbol * ELEMENTS_PER_SYMBOL;
        dim3 grid((num_elements + BLOCKSIZE - 1) / BLOCKSIZE);
        pw_multiply_half_kernel<<<grid, block, 0, stream>>>(
            d_X + offset, d_H + offset, d_Y + offset, num_elements);
    }
    gpuCheck(cudaEventRecord(stop, stream), "record");
    gpuCheck(cudaEventSynchronize(stop), "sync");
    gpuCheck(cudaEventElapsedTime(&results.kernel_exec_time, start, stop), "get time");
    
    // D2H Transfer
    gpuCheck(cudaEventRecord(start, stream), "record");
    gpuCheck(cudaMemcpyAsync(h_Y, d_Y, total_bytes, cudaMemcpyDeviceToHost, stream), "D2H Y");
    gpuCheck(cudaEventRecord(stop, stream), "record");
    gpuCheck(cudaEventSynchronize(stop), "sync");
    gpuCheck(cudaEventElapsedTime(&results.d2h_time, start, stop), "get time");
    
    results.total_time = results.h2d_time + 
                        (results.kernel_launch_time * KERNELS_PER_LAUNCH_LOOP) + 
                        results.kernel_exec_time + 
                        results.d2h_time;
    
    gpuCheck(cudaEventDestroy(start), "destroy");
    gpuCheck(cudaEventDestroy(stop), "destroy");
    
    return results;
}

TimingResults benchmarkKernelOnlyGraph(ComplexType* h_X, ComplexType* h_H, ComplexType* h_Y,
                                      ComplexType* d_X, ComplexType* d_H, ComplexType* d_Y,
                                      size_t total_bytes, cudaStream_t stream) {
    TimingResults results = {0};
    cudaEvent_t start, stop;
    gpuCheck(cudaEventCreate(&start), "create event");
    gpuCheck(cudaEventCreate(&stop), "create event");
    
    printf("\n--- Benchmarking Kernel-Only Graph ---\n");
    
    // H2D Transfer (same as standard)
    gpuCheck(cudaEventRecord(start, stream), "record");
    gpuCheck(cudaMemcpyAsync(d_X, h_X, total_bytes, cudaMemcpyHostToDevice, stream), "H2D X");
    gpuCheck(cudaMemcpyAsync(d_H, h_H, total_bytes, cudaMemcpyHostToDevice, stream), "H2D H");
    gpuCheck(cudaEventRecord(stop, stream), "record");
    gpuCheck(cudaEventSynchronize(stop), "sync");
    gpuCheck(cudaEventElapsedTime(&results.h2d_time, start, stop), "get time");
    
    // Create Graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    auto graph_start = std::chrono::high_resolution_clock::now();
    
    gpuCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
    
    dim3 block(BLOCKSIZE);
    int symbols_per_kernel = BATCH_SIZE_SYMBOLS / KERNELS_PER_LAUNCH_LOOP;
    for (int i = 0; i < KERNELS_PER_LAUNCH_LOOP; i++) {
        int start_symbol = i * symbols_per_kernel;
        int num_symbols = (i == KERNELS_PER_LAUNCH_LOOP - 1) ? 
            (BATCH_SIZE_SYMBOLS - start_symbol) : symbols_per_kernel;
        int num_elements = num_symbols * ELEMENTS_PER_SYMBOL;
        size_t offset = (size_t)start_symbol * ELEMENTS_PER_SYMBOL;
        dim3 grid((num_elements + BLOCKSIZE - 1) / BLOCKSIZE);
        pw_multiply_half_kernel<<<grid, block, 0, stream>>>(
            d_X + offset, d_H + offset, d_Y + offset, num_elements);
    }
    
    gpuCheck(cudaStreamEndCapture(stream, &graph), "end capture");
    gpuCheck(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0), "instantiate");
    
    auto graph_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> graph_duration = graph_end - graph_start;
    results.graph_creation_time = graph_duration.count();
    
    // Measure graph launch overhead
    results.kernel_launch_time = measureCpuLaunchOverhead(stream, 1); // Single graph launch
    results.per_kernel_launch_us = results.kernel_launch_time * 1000.0f / KERNELS_PER_LAUNCH_LOOP;
    
    // Warmup graph execution
    for (int i = 0; i < WARMUP_RUNS; i++) {
        gpuCheck(cudaGraphLaunch(graphExec, stream), "launch graph");
    }
    cudaStreamSynchronize(stream);
    
    // Time graph execution
    gpuCheck(cudaEventRecord(start, stream), "record");
    gpuCheck(cudaGraphLaunch(graphExec, stream), "launch graph");
    gpuCheck(cudaEventRecord(stop, stream), "record");
    gpuCheck(cudaEventSynchronize(stop), "sync");
    gpuCheck(cudaEventElapsedTime(&results.kernel_exec_time, start, stop), "get time");
    
    // D2H Transfer
    gpuCheck(cudaEventRecord(start, stream), "record");
    gpuCheck(cudaMemcpyAsync(h_Y, d_Y, total_bytes, cudaMemcpyDeviceToHost, stream), "D2H Y");
    gpuCheck(cudaEventRecord(stop, stream), "record");
    gpuCheck(cudaEventSynchronize(stop), "sync");
    gpuCheck(cudaEventElapsedTime(&results.d2h_time, start, stop), "get time");
    
    results.total_time = results.h2d_time + 
                        results.kernel_launch_time + // Single graph launch
                        results.kernel_exec_time + 
                        results.d2h_time;
    
    // Cleanup
    gpuCheck(cudaGraphExecDestroy(graphExec), "destroy exec");
    gpuCheck(cudaGraphDestroy(graph), "destroy graph");
    gpuCheck(cudaEventDestroy(start), "destroy");
    gpuCheck(cudaEventDestroy(stop), "destroy");
    
    return results;
}

/* ------------------------------------------------------------------ */
/*                                M A I N                             */
/* ------------------------------------------------------------------ */

int main() {
    printf("=== Detailed Phase Analysis: Standard Launch vs. Kernel-Only Graph ===\n");
    printf("Configuration:\n");
    printf("  Total Rays: %d\n", TOTAL_RAYS);
    printf("  Symbols per Ray: %d\n", SYMBOLS_PER_RAY);
    printf("  Elements per Symbol: %d\n", ELEMENTS_PER_SYMBOL);
    printf("  Kernels per Launch: %d\n", KERNELS_PER_LAUNCH_LOOP);
    printf("  Total Data Size: %.2f GB\n\n", 
           (double)BATCH_SIZE_SYMBOLS * ELEMENTS_PER_SYMBOL * sizeof(ComplexType) / 1e9);
    
    // Setup
    size_t total_bytes = (size_t)BATCH_SIZE_SYMBOLS * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
    ComplexType *h_X = safeHostAlloc(total_bytes, "h_X");
    ComplexType *h_H = safeHostAlloc(total_bytes, "h_H");
    ComplexType *h_Y = safeHostAlloc(total_bytes, "h_Y");

    printf("Initializing data...\n");
    for (int i = 0; i < BATCH_SIZE_SYMBOLS; i++) {
        auto vec_X = randomComplexHalfVector(ELEMENTS_PER_SYMBOL, i * 2);
        auto vec_H = randomComplexHalfVector(ELEMENTS_PER_SYMBOL, i * 2 + 1);
        size_t offset = (size_t)i * ELEMENTS_PER_SYMBOL;
        memcpy(&h_X[offset], vec_X.data(), ELEMENTS_PER_SYMBOL * sizeof(ComplexType));
        memcpy(&h_H[offset], vec_H.data(), ELEMENTS_PER_SYMBOL * sizeof(ComplexType));
    }
    
    ComplexType *d_X, *d_H, *d_Y;
    gpuCheck(cudaMalloc(&d_X, total_bytes), "malloc d_X");
    gpuCheck(cudaMalloc(&d_H, total_bytes), "malloc d_H");
    gpuCheck(cudaMalloc(&d_Y, total_bytes), "malloc d_Y");

    cudaStream_t stream;
    gpuCheck(cudaStreamCreate(&stream), "create stream");

    // Run benchmarks
    TimingResults standardResults = benchmarkStandardLaunch(h_X, h_H, h_Y, d_X, d_H, d_Y, total_bytes, stream);
    TimingResults graphResults = benchmarkKernelOnlyGraph(h_X, h_H, h_Y, d_X, d_H, d_Y, total_bytes, stream);

    // Display results
    printf("\n\n📊 DETAILED PERFORMANCE BREAKDOWN\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("                          Standard Launch    Kernel-Only Graph    Speedup\n");
    printf("                          ───────────────    ─────────────────    ───────\n");
    printf("H2D Transfer:             %8.2f ms        %8.2f ms         %.2fx\n", 
           standardResults.h2d_time, graphResults.h2d_time, 
           standardResults.h2d_time / graphResults.h2d_time);
    
    printf("Kernel Launch Overhead:   %8.3f ms        %8.3f ms         %.2fx\n", 
           standardResults.kernel_launch_time * KERNELS_PER_LAUNCH_LOOP, 
           graphResults.kernel_launch_time,
           (standardResults.kernel_launch_time * KERNELS_PER_LAUNCH_LOOP) / graphResults.kernel_launch_time);
    
    printf("Kernel Execution:         %8.3f ms        %8.3f ms         %.2fx\n", 
           standardResults.kernel_exec_time, graphResults.kernel_exec_time,
           standardResults.kernel_exec_time / graphResults.kernel_exec_time);
    
    printf("D2H Transfer:             %8.2f ms        %8.2f ms         %.2fx\n", 
           standardResults.d2h_time, graphResults.d2h_time,
           standardResults.d2h_time / graphResults.d2h_time);
    
    printf("───────────────────────────────────────────────────────────────\n");
    printf("TOTAL TIME:               %8.2f ms        %8.2f ms         %.2fx\n", 
           standardResults.total_time, graphResults.total_time,
           standardResults.total_time / graphResults.total_time);
    
    printf("\n\n📈 LAUNCH OVERHEAD ANALYSIS\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Standard Launch:\n");
    printf("  Total kernels:              %d\n", KERNELS_PER_LAUNCH_LOOP);
    printf("  Launch overhead per kernel: %.2f μs\n", standardResults.per_kernel_launch_us);
    printf("  Total launch overhead:      %.2f μs\n", 
           standardResults.per_kernel_launch_us * KERNELS_PER_LAUNCH_LOOP);
    
    printf("\nKernel-Only Graph:\n");
    printf("  Graph creation time:        %.2f ms (one-time cost)\n", graphResults.graph_creation_time);
    printf("  Graph launch overhead:      %.2f μs\n", graphResults.kernel_launch_time * 1000);
    printf("  Effective per-kernel cost:  %.2f μs\n", graphResults.per_kernel_launch_us);
    
    printf("\n🎯 Launch Overhead Reduction:  %.1fx\n", 
           standardResults.per_kernel_launch_us / graphResults.per_kernel_launch_us);
    
    printf("\n💡 INSIGHTS:\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    float launch_overhead_savings = (standardResults.kernel_launch_time * KERNELS_PER_LAUNCH_LOOP - 
                                    graphResults.kernel_launch_time);
    float percentage_savings = (launch_overhead_savings / standardResults.total_time) * 100;
    
    printf("• Launch overhead savings: %.3f ms (%.1f%% of total time)\n", 
           launch_overhead_savings, percentage_savings);
    
    if (percentage_savings < 1.0) {
        printf("• ⚠️  Launch overhead is minimal - graphs provide limited benefit\n");
        printf("• Consider using graphs when you have many small kernels\n");
    } else if (percentage_savings < 5.0) {
        printf("• Launch overhead is moderate - graphs provide some benefit\n");
    } else {
        printf("• ✅ Launch overhead is significant - graphs provide substantial benefit\n");
    }
    
    float compute_percentage = (standardResults.kernel_exec_time / standardResults.total_time) * 100;
    float transfer_percentage = ((standardResults.h2d_time + standardResults.d2h_time) / 
                                standardResults.total_time) * 100;
    
    printf("\n• Time distribution:\n");
    printf("  - Memory transfers: %.1f%%\n", transfer_percentage);
    printf("  - Kernel execution: %.1f%%\n", compute_percentage);
    printf("  - Launch overhead:  %.1f%%\n", 100 - transfer_percentage - compute_percentage);
    
    if (transfer_percentage > 50) {
        printf("\n• 📌 Memory transfers dominate - consider:\n");
        printf("  - Using larger batch sizes\n");
        printf("  - Overlapping transfers with computation\n");
        printf("  - Using GPUDirect for multi-GPU setups\n");
    }

    // Cleanup
    gpuCheck(cudaStreamDestroy(stream), "destroy stream");
    gpuCheck(cudaFree(d_X), "free");
    gpuCheck(cudaFree(d_H), "free");
    gpuCheck(cudaFree(d_Y), "free");
    safeFree(h_X);
    safeFree(h_H);
    safeFree(h_Y);

    printf("\nBenchmark completed successfully!\n");
    return 0;
}