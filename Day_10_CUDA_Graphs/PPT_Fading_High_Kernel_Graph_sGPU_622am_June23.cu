// Filename: benchmark_config_sweep.cu
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

constexpr int ELEMENTS_PER_SYMBOL = 4096;
constexpr int SYMBOLS_PER_RAY = 20;
constexpr int BLOCKSIZE = 128;
constexpr int WARMUP_RUNS = 5;

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
    float kernel_launch_time_ms;
    float kernel_exec_time_ms;
    float d2h_time_ms;
    float total_time_ms;
    float graph_creation_time_ms;
    float launch_overhead_per_kernel_us;
    float kernel_exec_per_kernel_us;
};

struct ComparisonResults {
    TestConfig config;
    BenchmarkResults standard;
    BenchmarkResults graph;
    float launch_speedup;
    float total_speedup;
    float launch_overhead_percentage;
};

/* ------------------------------------------------------------------ */
/*                       BENCHMARK FUNCTIONS                          */
/* ------------------------------------------------------------------ */

float measureCpuLaunchOverhead(cudaStream_t stream, int num_launches) {
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
    return duration.count() / num_launches;
}

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
    
    // Measure launch overhead
    results.kernel_launch_time_ms = measureCpuLaunchOverhead(stream, config.kernels_per_loop);
    results.launch_overhead_per_kernel_us = results.kernel_launch_time_ms * 1000.0f;
    
    // Kernel Execution
    dim3 block(BLOCKSIZE);
    int symbols_per_kernel = config.symbols_per_kernel();
    
    // Warmup
    for (int warm = 0; warm < WARMUP_RUNS; warm++) {
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
    }
    cudaStreamSynchronize(stream);
    
    // Time execution
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
    gpuCheck(cudaEventElapsedTime(&results.kernel_exec_time_ms, start, stop), "get time");
    
    results.kernel_exec_per_kernel_us = (results.kernel_exec_time_ms * 1000.0f) / config.kernels_per_loop;
    
    // D2H Transfer
    gpuCheck(cudaEventRecord(start, stream), "record");
    gpuCheck(cudaMemcpyAsync(h_Y, d_Y, total_bytes, cudaMemcpyDeviceToHost, stream), "D2H Y");
    gpuCheck(cudaEventRecord(stop, stream), "record");
    gpuCheck(cudaEventSynchronize(stop), "sync");
    gpuCheck(cudaEventElapsedTime(&results.d2h_time_ms, start, stop), "get time");
    
    results.total_time_ms = results.h2d_time_ms + 
                           (results.kernel_launch_time_ms * config.kernels_per_loop) + 
                           results.kernel_exec_time_ms + 
                           results.d2h_time_ms;
    
    gpuCheck(cudaEventDestroy(start), "destroy");
    gpuCheck(cudaEventDestroy(stop), "destroy");
    
    return results;
}

BenchmarkResults benchmarkGraphLaunch(const TestConfig& config,
                                     ComplexType* h_X, ComplexType* h_H, ComplexType* h_Y,
                                     ComplexType* d_X, ComplexType* d_H, ComplexType* d_Y,
                                     cudaStream_t stream) {
    BenchmarkResults results = {0};
    cudaEvent_t start, stop;
    gpuCheck(cudaEventCreate(&start), "create event");
    gpuCheck(cudaEventCreate(&stop), "create event");
    
    size_t total_bytes = config.total_bytes();
    
    // H2D Transfer (same as standard)
    gpuCheck(cudaEventRecord(start, stream), "record");
    gpuCheck(cudaMemcpyAsync(d_X, h_X, total_bytes, cudaMemcpyHostToDevice, stream), "H2D X");
    gpuCheck(cudaMemcpyAsync(d_H, h_H, total_bytes, cudaMemcpyHostToDevice, stream), "H2D H");
    gpuCheck(cudaEventRecord(stop, stream), "record");
    gpuCheck(cudaEventSynchronize(stop), "sync");
    gpuCheck(cudaEventElapsedTime(&results.h2d_time_ms, start, stop), "get time");
    
    // Create Graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    auto graph_start = std::chrono::high_resolution_clock::now();
    
    gpuCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
    
    dim3 block(BLOCKSIZE);
    int symbols_per_kernel = config.symbols_per_kernel();
    
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
    
    gpuCheck(cudaStreamEndCapture(stream, &graph), "end capture");
    gpuCheck(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0), "instantiate");
    
    auto graph_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> graph_duration = graph_end - graph_start;
    results.graph_creation_time_ms = graph_duration.count();
    
    // Measure graph launch overhead
    results.kernel_launch_time_ms = measureCpuLaunchOverhead(stream, 1);
    results.launch_overhead_per_kernel_us = (results.kernel_launch_time_ms * 1000.0f) / config.kernels_per_loop;
    
    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        gpuCheck(cudaGraphLaunch(graphExec, stream), "launch graph");
    }
    cudaStreamSynchronize(stream);
    
    // Time execution
    gpuCheck(cudaEventRecord(start, stream), "record");
    gpuCheck(cudaGraphLaunch(graphExec, stream), "launch graph");
    gpuCheck(cudaEventRecord(stop, stream), "record");
    gpuCheck(cudaEventSynchronize(stop), "sync");
    gpuCheck(cudaEventElapsedTime(&results.kernel_exec_time_ms, start, stop), "get time");
    
    results.kernel_exec_per_kernel_us = (results.kernel_exec_time_ms * 1000.0f) / config.kernels_per_loop;
    
    // D2H Transfer
    gpuCheck(cudaEventRecord(start, stream), "record");
    gpuCheck(cudaMemcpyAsync(h_Y, d_Y, total_bytes, cudaMemcpyDeviceToHost, stream), "D2H Y");
    gpuCheck(cudaEventRecord(stop, stream), "record");
    gpuCheck(cudaEventSynchronize(stop), "sync");
    gpuCheck(cudaEventElapsedTime(&results.d2h_time_ms, start, stop), "get time");
    
    results.total_time_ms = results.h2d_time_ms + 
                           results.kernel_launch_time_ms + 
                           results.kernel_exec_time_ms + 
                           results.d2h_time_ms;
    
    // Cleanup
    gpuCheck(cudaGraphExecDestroy(graphExec), "destroy exec");
    gpuCheck(cudaGraphDestroy(graph), "destroy graph");
    gpuCheck(cudaEventDestroy(start), "destroy");
    gpuCheck(cudaEventDestroy(stop), "destroy");
    
    return results;
}

ComparisonResults runComparison(const TestConfig& config) {
    ComparisonResults comparison;
    comparison.config = config;
    
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("Testing: %s\n", config.description.c_str());
    printf("  Total rays: %d, Kernels: %d, Rays/kernel: %d\n", 
           config.total_rays, config.kernels_per_loop, config.rays_per_kernel());
    
    // Allocate memory
    size_t total_bytes = config.total_bytes();
    ComplexType *h_X = safeHostAlloc(total_bytes, "h_X");
    ComplexType *h_H = safeHostAlloc(total_bytes, "h_H");
    ComplexType *h_Y = safeHostAlloc(total_bytes, "h_Y");
    
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
    comparison.standard = benchmarkStandardLaunch(config, h_X, h_H, h_Y, d_X, d_H, d_Y, stream);
    comparison.graph = benchmarkGraphLaunch(config, h_X, h_H, h_Y, d_X, d_H, d_Y, stream);
    
    // Calculate speedups
    comparison.launch_speedup = (comparison.standard.kernel_launch_time_ms * config.kernels_per_loop) / 
                               comparison.graph.kernel_launch_time_ms;
    comparison.total_speedup = comparison.standard.total_time_ms / comparison.graph.total_time_ms;
    comparison.launch_overhead_percentage = 
        ((comparison.standard.kernel_launch_time_ms * config.kernels_per_loop) / 
         comparison.standard.total_time_ms) * 100.0f;
    
    // Cleanup
    gpuCheck(cudaStreamDestroy(stream), "destroy");
    gpuCheck(cudaFree(d_X), "free");
    gpuCheck(cudaFree(d_H), "free");
    gpuCheck(cudaFree(d_Y), "free");
    safeFree(h_X);
    safeFree(h_H);
    safeFree(h_Y);
    
    return comparison;
}

/* ------------------------------------------------------------------ */
/*                         OUTPUT FUNCTIONS                           */
/* ------------------------------------------------------------------ */

void printResults(const std::vector<ComparisonResults>& results) {
    printf("\n\n📊 CONFIGURATION SWEEP RESULTS\n");
    printf("════════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("%-30s | Kernels | Launch OH | Exec Time | Total Time | Graph Speedup | OH %%\n", "Configuration");
    printf("%-30s | ------- | --------- | --------- | ---------- | ------------- | -----\n", 
           "------------------------------");
    
    for (const auto& r : results) {
        printf("%-30s | %7d | %7.2f ms | %7.2f ms | %8.1f ms | %11.1fx | %5.1f%%\n",
               r.config.description.c_str(),
               r.config.kernels_per_loop,
               r.standard.kernel_launch_time_ms * r.config.kernels_per_loop,
               r.standard.kernel_exec_time_ms,
               r.standard.total_time_ms,
               r.launch_speedup,
               r.launch_overhead_percentage);
    }
    
    printf("\n📈 DETAILED PER-KERNEL METRICS\n");
    printf("════════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("%-30s | Standard Launch/kernel | Graph Launch/kernel | Exec Time/kernel | Graph Creation\n", 
           "Configuration");
    printf("%-30s | --------------------- | ------------------ | ---------------- | --------------\n",
           "------------------------------");
    
    for (const auto& r : results) {
        printf("%-30s | %18.2f μs | %15.2f μs | %13.2f μs | %11.1f ms\n",
               r.config.description.c_str(),
               r.standard.launch_overhead_per_kernel_us,
               r.graph.launch_overhead_per_kernel_us,
               r.standard.kernel_exec_per_kernel_us,
               r.graph.graph_creation_time_ms);
    }
}

void saveResultsToCSV(const std::vector<ComparisonResults>& results, const std::string& filename) {
    std::ofstream file(filename);
    
    // Header
    file << "Description,Total_Rays,Kernels,Rays_Per_Kernel,"
         << "Standard_H2D_ms,Standard_Launch_ms,Standard_Exec_ms,Standard_D2H_ms,Standard_Total_ms,"
         << "Graph_H2D_ms,Graph_Launch_ms,Graph_Exec_ms,Graph_D2H_ms,Graph_Total_ms,"
         << "Graph_Creation_ms,Launch_Speedup,Total_Speedup,Launch_Overhead_Percent\n";
    
    // Data
    for (const auto& r : results) {
        file << r.config.description << ","
             << r.config.total_rays << ","
             << r.config.kernels_per_loop << ","
             << r.config.rays_per_kernel() << ","
             << r.standard.h2d_time_ms << ","
             << r.standard.kernel_launch_time_ms * r.config.kernels_per_loop << ","
             << r.standard.kernel_exec_time_ms << ","
             << r.standard.d2h_time_ms << ","
             << r.standard.total_time_ms << ","
             << r.graph.h2d_time_ms << ","
             << r.graph.kernel_launch_time_ms << ","
             << r.graph.kernel_exec_time_ms << ","
             << r.graph.d2h_time_ms << ","
             << r.graph.total_time_ms << ","
             << r.graph.graph_creation_time_ms << ","
             << r.launch_speedup << ","
             << r.total_speedup << ","
             << r.launch_overhead_percentage << "\n";
    }
    
    file.close();
    printf("\n💾 Results saved to %s\n", filename.c_str());
}

/* ------------------------------------------------------------------ */
/*                                M A I N                             */
/* ------------------------------------------------------------------ */

int main() {
    printf("🚀 CUDA Graph Configuration Sweep Test\n");
    printf("=====================================\n");
    
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
    printf("SM Count: %d, Max Threads/Block: %d\n\n", prop.multiProcessorCount, prop.maxThreadsPerBlock);
    
    // Define test configurations
    std::vector<TestConfig> configs = {
        // Kernel granularity tests (fixed workload of 4000 rays)
        {4000, 1,     "Single mega-kernel"},
        {4000, 4,     "Few large kernels (baseline)"},
        {4000, 10,    "10 large kernels"},
        {4000, 40,    "40 medium kernels (100 rays)"},
        {4000, 100,   "100 medium kernels (40 rays)"},
        {4000, 400,   "400 small kernels (10 rays)"},
        {4000, 1000,  "1000 small kernels (4 rays)"},
        {4000, 4000,  "4000 tiny kernels (1 ray)"},
        {4000, 20000, "20K micro kernels (5 symbols)"},
        {4000, 40000, "40K micro kernels (2 symbols)"},  // Still aggressive but more realistic
        {4000, 80000, "80K nano kernels (1 symbol)"},
        {4000, 160000, "160K tiny kernels (0.5 symbol)"}, // Pushing limits
        
        // Workload scaling tests (1 ray per kernel)
        {100,   100,   "Small workload (100 rays)"},
        {500,   500,   "Small-medium (500 rays)"},
        {1000,  1000,  "Medium workload (1000 rays)"},
        {2000,  2000,  "Medium-large (2000 rays)"},
        {8000,  8000,  "Large workload (8000 rays)"},
        {10000, 10000, "XL workload (10000 rays)"},
        
        // Special scenarios
        {4000, 8,     "8 kernels (multi-GPU sim)"},
        {4000, 20,    "20 kernels (streaming)"},
        {4000, 200,   "200 kernels (balanced)"},
    };
    
    // Run all tests
    std::vector<ComparisonResults> results;
    for (const auto& config : configs) {
        try {
            results.push_back(runComparison(config));
        } catch (const std::exception& e) {
            printf("Error testing config %s: %s\n", config.description.c_str(), e.what());
        }
    }
    
    // Display and save results
    printResults(results);
    saveResultsToCSV(results, "cuda_graph_sweep_results.csv");
    
    // Find optimal configuration
    printf("\n\n🏆 ANALYSIS SUMMARY\n");
    printf("════════════════════════════════════════════════════════════════\n");
    
    // Find best speedup
    auto best_speedup = std::max_element(results.begin(), results.end(),
        [](const ComparisonResults& a, const ComparisonResults& b) {
            return a.launch_speedup < b.launch_speedup;
        });
    
    printf("Best launch speedup: %.1fx with configuration '%s'\n",
           best_speedup->launch_speedup, best_speedup->config.description.c_str());
    
    // Find break-even point
    for (const auto& r : results) {
        if (r.config.total_rays == 4000 && r.launch_overhead_percentage > 1.0) {
            printf("Break-even point: ~%d kernels (%.1f%% launch overhead)\n",
                   r.config.kernels_per_loop, r.launch_overhead_percentage);
            break;
        }
    }
    
    // Memory vs compute bound analysis
    float avg_memory_percentage = 0;
    for (const auto& r : results) {
        float memory_time = r.standard.h2d_time_ms + r.standard.d2h_time_ms;
        avg_memory_percentage += (memory_time / r.standard.total_time_ms) * 100.0f;
    }
    avg_memory_percentage /= results.size();
    
    printf("\nWorkload characteristics:\n");
    printf("  Average memory transfer time: %.1f%%\n", avg_memory_percentage);
    printf("  %s\n", avg_memory_percentage > 50 ? "❗ Memory-bound workload" : "✅ Compute-bound workload");
    
    printf("\n✅ Configuration sweep completed successfully!\n");
    
    return 0;
}