#include <cstdio>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <random>
#include <vector>
#include <string>
#include <algorithm> // For std::min
#include <nvtx3/nvToolsExt.h>  // NVTX for profiling markers
#include <thread> // For parallel graph launch

constexpr int ELEMENTS_PER_SYMBOL = 4096;
constexpr int SYMBOLS_PER_RAY = 20;
constexpr int TOTAL_RAYS = 5000;             // Total workload size
constexpr int BLOCKSIZE = 128;
constexpr int STREAMS_PER_GPU = 64;          // Kernels per GPU to capture in graph

// The entire workload is processed in one batch
constexpr int BATCH_SIZE_SYMBOLS = TOTAL_RAYS * SYMBOLS_PER_RAY;

using ComplexType = __half2;
const char* precision_name = "Half-Precision (FP16)";

/* ------------------------------------------------------------------ */
/*                            K E R N E L S                           */
/* ------------------------------------------------------------------ */

__global__ void pw_multiply_half_kernel(const __half2* __restrict__ X,
                                        const __half2* __restrict__ H,
                                        __half2*       __restrict__ Y,
                                        int                         n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        __half2 x_val = X[i];
        __half2 h_val = H[i];
        __half x_real = __low2half(x_val);
        __half x_imag = __high2half(x_val);
        __half h_real = __low2half(h_val);
        __half h_imag = __high2half(h_val);
        __half result_real = __hsub(__hmul(x_real, h_real), __hmul(x_imag, h_imag));
        __half result_imag = __hadd(__hmul(x_real, h_imag), __hmul(x_imag, h_real));
        Y[i] = __halves2half2(result_real, result_imag);
    }
}

/* ------------------------------------------------------------------ */
/*                              H E L P E R S                         */
/* ------------------------------------------------------------------ */

void gpuCheck(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

std::vector<__half2> randomComplexHalfVector(int n, int seed = 0xC0FFEE)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<__half2> v(n);
    for (auto& c : v) {
        float real = dist(gen);
        float imag = dist(gen);
        c = __halves2half2(__float2half(real), __float2half(imag));
    }
    return v;
}

ComplexType* safeHostAlloc(size_t bytes, const char* name) {
    ComplexType* ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (err == cudaSuccess) {
        printf("  %s: cudaHostAlloc success (%.2f GB)\n", name, bytes / (1024.0f * 1024.0f * 1024.0f));
        return ptr;
    }
    ptr = (ComplexType*)malloc(bytes);
    if (ptr) { printf("  %s: malloc fallback success (%.2f GB)\n", name, bytes / (1024.0f * 1024.0f * 1024.0f)); return ptr; }
    printf("  %s: FAILED to allocate %.2f GB\n", name, bytes / (1024.0f * 1024.0f * 1024.0f));
    return nullptr;
}

void safeFree(ComplexType* ptr) {
    if (ptr) { if (cudaFreeHost(ptr) != cudaSuccess) { free(ptr); } }
}

struct TimingResults {
    float kernel_launch_time;
    float total_time;
    int num_graph_launches;
};

/* ------------------------------------------------------------------ */
/*                                M A I N                             */
/* ------------------------------------------------------------------ */

int main()
{
    nvtxRangePush("Program_Initialization");
    int deviceCount;
    gpuCheck(cudaGetDeviceCount(&deviceCount), "get device count");
    if (deviceCount < 1) { printf("No CUDA devices found!\n"); return 1; }
    printf("=== Multi-GPU Point-wise Complex Multiplication with CUDA Graphs ===\n");
    printf("Found %d CUDA device(s)\n", deviceCount);
    long long totalOperations = (long long)TOTAL_RAYS * SYMBOLS_PER_RAY * ELEMENTS_PER_SYMBOL;
    printf("Workload: %lld operations (%.1f million)\n", totalOperations, totalOperations / 1000000.0f);
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        gpuCheck(cudaGetDeviceProperties(&prop, dev), "get device properties");
        printf("\nDevice %d: %s\n", dev, prop.name);
    }
    printf("\n");
    nvtxRangePop();

    nvtxRangePush("Host_Memory_And_Data_Generation");
    size_t totalBytes = (size_t)BATCH_SIZE_SYMBOLS * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
    printf("Allocating %.2f GB of pinned host memory...\n", (totalBytes * 3) / (1024.0f * 1024.0f * 1024.0f));
    ComplexType *batch_X = safeHostAlloc(totalBytes, "batch_X");
    ComplexType *batch_H = safeHostAlloc(totalBytes, "batch_H");
    ComplexType *batch_Y = safeHostAlloc(totalBytes, "batch_Y");
    if (!batch_X || !batch_H || !batch_Y) return 1;
    printf("Generating test data...\n");
    for (int i = 0; i < BATCH_SIZE_SYMBOLS; i++) {
        auto vec_X = randomComplexHalfVector(ELEMENTS_PER_SYMBOL, i * 2 + 1000);
        auto vec_H = randomComplexHalfVector(ELEMENTS_PER_SYMBOL, i * 2 + 2000);
        memcpy(&batch_X[(size_t)i * ELEMENTS_PER_SYMBOL], vec_X.data(), (size_t)ELEMENTS_PER_SYMBOL * sizeof(ComplexType));
        memcpy(&batch_H[(size_t)i * ELEMENTS_PER_SYMBOL], vec_H.data(), (size_t)ELEMENTS_PER_SYMBOL * sizeof(ComplexType));
    }
    nvtxRangePop();

    // ==============================================
    // SINGLE GPU GRAPH PERFORMANCE
    // ==============================================
    TimingResults singleGpuResults = {0};
    {
        nvtxRangePush("Single_GPU_Graph_Benchmark");
        printf("\n=== SINGLE GPU PERFORMANCE (with CUDA Graph) ===\n");
        
        ComplexType *d_X, *d_H, *d_Y;
        gpuCheck(cudaSetDevice(0), "set device 0");
        gpuCheck(cudaMalloc(&d_X, totalBytes), "malloc X single");
        gpuCheck(cudaMalloc(&d_H, totalBytes), "malloc H single");
        gpuCheck(cudaMalloc(&d_Y, totalBytes), "malloc Y single");
        
        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        gpuCheck(cudaStreamCreate(&stream), "create stream");
        
        printf("  Phase 1: Capturing and Instantiating Graph for 1 GPU...\n");
        gpuCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
        gpuCheck(cudaMemcpyAsync(d_X, batch_X, totalBytes, cudaMemcpyHostToDevice, stream), "capture copy X");
        gpuCheck(cudaMemcpyAsync(d_H, batch_H, totalBytes, cudaMemcpyHostToDevice, stream), "capture copy H");
        dim3 block(BLOCKSIZE);
        int symbolsPerKernel = BATCH_SIZE_SYMBOLS / STREAMS_PER_GPU;
        for (int s = 0; s < STREAMS_PER_GPU; s++) {
            int chunkStartSymbol = s * symbolsPerKernel;
            int chunkElements = symbolsPerKernel * ELEMENTS_PER_SYMBOL;
            if (s == STREAMS_PER_GPU - 1) chunkElements = (BATCH_SIZE_SYMBOLS - chunkStartSymbol) * ELEMENTS_PER_SYMBOL;
            if (chunkElements <= 0) continue;
            size_t offset = (size_t)chunkStartSymbol * ELEMENTS_PER_SYMBOL;
            dim3 grid((chunkElements + BLOCKSIZE - 1) / BLOCKSIZE);
            pw_multiply_half_kernel<<<grid, block, 0, stream>>>(
                (ComplexType*)((char*)d_X + offset), (ComplexType*)((char*)d_H + offset), (ComplexType*)((char*)d_Y + offset), chunkElements);
        }
        gpuCheck(cudaGetLastError(), "kernel launch capture");
        gpuCheck(cudaMemcpyAsync(batch_Y, d_Y, totalBytes, cudaMemcpyDeviceToHost, stream), "capture copy Y");
        gpuCheck(cudaStreamEndCapture(stream, &graph), "end capture");
        gpuCheck(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0), "instantiate graph");

        printf("  Phase 2: Launching Graph and Synchronizing...\n");
        cudaEvent_t start, stop;
        gpuCheck(cudaEventCreate(&start), "create start");
        gpuCheck(cudaEventCreate(&stop), "create stop");
        gpuCheck(cudaEventRecord(start, stream), "record start");
        gpuCheck(cudaGraphLaunch(graphExec, stream), "launch graph");
        gpuCheck(cudaEventRecord(stop, stream), "record stop");
        gpuCheck(cudaEventSynchronize(stop), "sync stop event");
        gpuCheck(cudaEventElapsedTime(&singleGpuResults.total_time, start, stop), "get total gpu time");
        singleGpuResults.num_graph_launches = 1;

        gpuCheck(cudaFree(d_X), "free X single"); gpuCheck(cudaFree(d_H), "free H single"); gpuCheck(cudaFree(d_Y), "free Y single");
        gpuCheck(cudaGraphExecDestroy(graphExec), "destroy exec"); gpuCheck(cudaGraphDestroy(graph), "destroy graph"); gpuCheck(cudaStreamDestroy(stream), "destroy stream");
        gpuCheck(cudaEventDestroy(start), "destroy start"); gpuCheck(cudaEventDestroy(stop), "destroy stop");
        nvtxRangePop();
    }
    
    // ==============================================
    // MULTI-GPU GRAPH PERFORMANCE
    // ==============================================
    TimingResults multiGpuResults = {0};
    {
        nvtxRangePush("Multi_GPU_Graph_Benchmark");
        printf("\n=== MULTI-GPU PERFORMANCE (with Per-GPU CUDA Graphs) ===\n");

        std::vector<cudaGraph_t> graphs(deviceCount);
        std::vector<cudaGraphExec_t> graph_execs(deviceCount);
        std::vector<cudaStream_t> streams(deviceCount);

        std::vector<ComplexType*> dev_X(deviceCount), dev_H(deviceCount), dev_Y(deviceCount);

        printf("  Phase 1: Creating Per-GPU Graphs...\n");
        size_t symbols_processed = 0;
        int base_symbols = BATCH_SIZE_SYMBOLS / deviceCount;
        int remainder_symbols = BATCH_SIZE_SYMBOLS % deviceCount;

        for (int dev = 0; dev < deviceCount; ++dev) {
            nvtxRangePushA(("Setup_GPU_" + std::to_string(dev)).c_str());
            gpuCheck(cudaSetDevice(dev), "set device");
            gpuCheck(cudaStreamCreate(&streams[dev]), "create stream for dev");

            int gpu_symbols = base_symbols + (dev < remainder_symbols ? 1 : 0);
            size_t gpu_bytes = (size_t)gpu_symbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
            size_t host_offset_bytes = symbols_processed * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);

            gpuCheck(cudaMalloc(&dev_X[dev], gpu_bytes), "malloc X multi");
            gpuCheck(cudaMalloc(&dev_H[dev], gpu_bytes), "malloc H multi");
            gpuCheck(cudaMalloc(&dev_Y[dev], gpu_bytes), "malloc Y multi");

            gpuCheck(cudaStreamBeginCapture(streams[dev], cudaStreamCaptureModeGlobal), "begin capture multi");
            gpuCheck(cudaMemcpyAsync(dev_X[dev], batch_X + symbols_processed * ELEMENTS_PER_SYMBOL, gpu_bytes, cudaMemcpyHostToDevice, streams[dev]), "capture H2D X");
            gpuCheck(cudaMemcpyAsync(dev_H[dev], batch_H + symbols_processed * ELEMENTS_PER_SYMBOL, gpu_bytes, cudaMemcpyHostToDevice, streams[dev]), "capture H2D H");

            int symbols_per_kernel = gpu_symbols / STREAMS_PER_GPU;
            for (int s = 0; s < STREAMS_PER_GPU; s++) {
                int chunk_start_symbol_local = s * symbols_per_kernel;
                int chunk_symbols = (s == STREAMS_PER_GPU - 1) ? (gpu_symbols - chunk_start_symbol_local) : symbols_per_kernel;
                if (chunk_symbols <= 0) continue;
                int chunk_elements = chunk_symbols * ELEMENTS_PER_SYMBOL;
                size_t local_offset = (size_t)chunk_start_symbol_local * ELEMENTS_PER_SYMBOL;
                dim3 grid((chunk_elements + BLOCKSIZE - 1) / BLOCKSIZE), block(BLOCKSIZE);
                pw_multiply_half_kernel<<<grid, block, 0, streams[dev]>>>(
                    (ComplexType*)((char*)dev_X[dev] + local_offset), (ComplexType*)((char*)dev_H[dev] + local_offset),
                    (ComplexType*)((char*)dev_Y[dev] + local_offset), chunk_elements);
            }
            gpuCheck(cudaGetLastError(), "kernel launch capture multi");

            gpuCheck(cudaMemcpyAsync(batch_Y + symbols_processed * ELEMENTS_PER_SYMBOL, dev_Y[dev], gpu_bytes, cudaMemcpyDeviceToHost, streams[dev]), "capture D2H Y");
            gpuCheck(cudaStreamEndCapture(streams[dev], &graphs[dev]), "end capture multi");
            gpuCheck(cudaGraphInstantiate(&graph_execs[dev], graphs[dev], NULL, NULL, 0), "instantiate multi");

            symbols_processed += gpu_symbols;
            nvtxRangePop();
        }
        
        printf("  Phase 2: Launching all Graphs in Parallel...\n");
        cudaEvent_t start, stop;
        gpuCheck(cudaEventCreate(&start), "create start multi");
        gpuCheck(cudaEventCreate(&stop), "create stop multi");
        
        // Use a single event on the default stream to mark the beginning
        gpuCheck(cudaEventRecord(start, 0), "record start multi");

        std::vector<std::thread> launch_threads;
        for (int dev = 0; dev < deviceCount; ++dev) {
            launch_threads.emplace_back([=, &graph_execs, &streams] {
                gpuCheck(cudaSetDevice(dev), "thread set device");
                gpuCheck(cudaGraphLaunch(graph_execs[dev], streams[dev]), "launch graph multi");
            });
        }
        for (auto& t : launch_threads) { t.join(); }

        // Sync all streams on all devices
        for (int dev = 0; dev < deviceCount; ++dev) {
            gpuCheck(cudaSetDevice(dev), "set device for sync");
            gpuCheck(cudaStreamSynchronize(streams[dev]), "sync stream multi");
        }
        
        gpuCheck(cudaEventRecord(stop, 0), "record stop multi");
        gpuCheck(cudaEventSynchronize(stop), "sync stop event multi");
        gpuCheck(cudaEventElapsedTime(&multiGpuResults.total_time, start, stop), "get total gpu time multi");
        multiGpuResults.num_graph_launches = deviceCount;

        printf("Cleaning up multi-GPU resources...\n");
        for (int dev = 0; dev < deviceCount; ++dev) {
            gpuCheck(cudaSetDevice(dev), "set device for cleanup");
            gpuCheck(cudaFree(dev_X[dev]), "free X"); gpuCheck(cudaFree(dev_H[dev]), "free H"); gpuCheck(cudaFree(dev_Y[dev]), "free Y");
            gpuCheck(cudaGraphExecDestroy(graph_execs[dev]), "destroy exec"); gpuCheck(cudaGraphDestroy(graphs[dev]), "destroy graph");
            gpuCheck(cudaStreamDestroy(streams[dev]), "destroy stream");
        }
        gpuCheck(cudaEventDestroy(start), "destroy start"); gpuCheck(cudaEventDestroy(stop), "destroy stop");
        nvtxRangePop();
    }
    
    // ==============================================
    // DETAILED PERFORMANCE COMPARISON
    // ==============================================
    nvtxRangePush("Performance_Analysis");
    printf("\n=== DETAILED PERFORMANCE COMPARISON (Graph vs. Graph) ===\n");
    printf("\n📊 TIMING BREAKDOWN:\n");
    printf("                          Single GPU    Multi-GPU (%d)\n", deviceCount);
    printf("                          -----------    --------------\n");
    printf("Total Execution Time:   %8.2f ms      %8.2f ms\n", singleGpuResults.total_time, multiGpuResults.total_time);
    
    printf("\n📈 PERFORMANCE METRICS:\n");
    printf("Single GPU (with CUDA Graph):\n");
    printf("  Graph launches:         %d\n", singleGpuResults.num_graph_launches);
    printf("  Complex ops/second:     %.1f billion ops/s\n", (float)totalOperations / (singleGpuResults.total_time / 1000.0f) / 1000000000.0f);
    
    printf("\nMulti-GPU (with Per-GPU Graphs):\n");
    printf("  Graph launches:         %d\n", multiGpuResults.num_graph_launches);
    printf("  Complex ops/second:     %.1f billion ops/s\n", (float)totalOperations / (multiGpuResults.total_time / 1000.0f) / 1000000000.0f);
    
    printf("\n🚀 EFFICIENCY ANALYSIS:\n");
    float totalSpeedup = singleGpuResults.total_time / multiGpuResults.total_time;
    printf("Total Time Speedup: %.2fx\n", totalSpeedup);
    printf("Parallel Efficiency: %.1f%% (%.1f%% is ideal)\n", (totalSpeedup / deviceCount) * 100.0f, 100.0f);
    nvtxRangePop();
    
    // Final Cleanup
    nvtxRangePush("Cleanup");
    printf("\nCleaning up remaining memory...\n");
    safeFree(batch_X); safeFree(batch_H); safeFree(batch_Y);
    nvtxRangePop(); 
    
    printf("\nProcessing and analysis complete!\n");
    return 0;
}