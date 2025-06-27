#include <cstdio>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <random>
#include <vector>
#include <algorithm> // For std::min
#include <nvtx3/nvToolsExt.h>  // NVTX for profiling markers

constexpr int ELEMENTS_PER_SYMBOL = 4096;    // 4K elements per symbol
constexpr int SYMBOLS_PER_RAY = 20;          // 20 symbols per ray
constexpr int TOTAL_RAYS = 1000;             // 1K rays for testing
constexpr int BLOCKSIZE = 128;               // Threads per block
constexpr int STREAMS_PER_GPU = 64;          // Balanced for performance

// SAFE: Conservative batch size that should work in most Docker environments
constexpr int BATCH_SIZE_RAYS = 1000;        // Process 1K rays at a time (20K symbols)
constexpr int BATCH_SIZE_SYMBOLS = BATCH_SIZE_RAYS * SYMBOLS_PER_RAY;

// Memory optimization: Use half-precision for 50% memory savings
using ComplexType = __half2;  // 4 bytes instead of 8 bytes
const char* precision_name = "Half-Precision (FP16)";

/* ------------------------------------------------------------------ */
/*                            K E R N E L S                           */
/* ------------------------------------------------------------------ */

// Half-precision complex multiplication kernel
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
    if (ptr) {
        printf("  %s: malloc fallback success (%.2f GB)\n", name, bytes / (1024.0f * 1024.0f * 1024.0f));
        return ptr;
    }
    printf("  %s: FAILED to allocate %.2f GB\n", name, bytes / (1024.0f * 1024.0f * 1024.0f));
    return nullptr;
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
    float kernel_launch_time;
    float kernel_exec_time;
    float d2h_time;
    float total_time;
    int num_kernels_launched;
    int num_streams_used;
};

/* ------------------------------------------------------------------ */
/*                                M A I N                             */
/* ------------------------------------------------------------------ */

int main()
{
    nvtxRangePush("Program_Initialization");
    
    int deviceCount;
    gpuCheck(cudaGetDeviceCount(&deviceCount), "get device count");
    
    if (deviceCount < 1) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    printf("=== 1K Rays Multi-GPU Point-wise Complex Multiplication (%s) ===\n", precision_name);
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    long long totalOperations = (long long)TOTAL_RAYS * SYMBOLS_PER_RAY * ELEMENTS_PER_SYMBOL;
    printf("Workload: %lld operations (%.1f million)\n", 
           totalOperations, totalOperations / 1000000.0f);
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        gpuCheck(cudaGetDeviceProperties(&prop, dev), "get device properties");
        printf("\nDevice %d: %s\n", dev, prop.name);
    }
    printf("\n");

    nvtxRangePop(); // Program_Initialization

    // Host Memory Allocation and Data Generation
    nvtxRangePush("Host_Memory_Allocation");
    
    size_t batchBytes = (size_t)BATCH_SIZE_SYMBOLS * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
    
    printf("Allocating %.2f GB of pinned host memory...\n", (batchBytes * 3) / (1024.0f * 1024.0f * 1024.0f));
    ComplexType *batch_X = safeHostAlloc(batchBytes, "batch_X");
    ComplexType *batch_H = safeHostAlloc(batchBytes, "batch_H");
    ComplexType *batch_Y = safeHostAlloc(batchBytes, "batch_Y");
    if (!batch_X || !batch_H || !batch_Y) return 1;

    nvtxRangePop(); // Host_Memory_Allocation

    nvtxRangePush("Data_Generation");
    printf("Generating test data...\n");
    for (int i = 0; i < BATCH_SIZE_SYMBOLS; i++) {
        auto vec_X = randomComplexHalfVector(ELEMENTS_PER_SYMBOL, i * 2 + 1000);
        auto vec_H = randomComplexHalfVector(ELEMENTS_PER_SYMBOL, i * 2 + 2000);
        memcpy(&batch_X[(size_t)i * ELEMENTS_PER_SYMBOL], vec_X.data(), (size_t)ELEMENTS_PER_SYMBOL * sizeof(ComplexType));
        memcpy(&batch_H[(size_t)i * ELEMENTS_PER_SYMBOL], vec_H.data(), (size_t)ELEMENTS_PER_SYMBOL * sizeof(ComplexType));
    }
    nvtxRangePop(); // Data_Generation

    // ==============================================
    // SINGLE GPU BASELINE PERFORMANCE (Corrected with CUDA Graph)
    // ==============================================
    nvtxRangePush("Single_GPU_Performance");
    printf("\n=== SINGLE GPU PERFORMANCE (with CUDA Graph) ===\n");
    
    nvtxRangePush("Single_GPU_Memory_Allocation");
    ComplexType *dev_X_single, *dev_H_single, *dev_Y_single;
    gpuCheck(cudaSetDevice(0), "set device 0");
    gpuCheck(cudaMalloc(&dev_X_single, batchBytes), "malloc X single");
    gpuCheck(cudaMalloc(&dev_H_single, batchBytes), "malloc H single");
    gpuCheck(cudaMalloc(&dev_Y_single, batchBytes), "malloc Y single");
    
    cudaStream_t captureStream;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    gpuCheck(cudaStreamCreate(&captureStream), "create capture stream");
    nvtxRangePop(); // Single_GPU_Memory_Allocation
    
    TimingResults singleGpuResults = {0};
    int captured_kernels = 0;
    
    nvtxRangePush("Single_GPU_CUDA_Graph_Creation");
    printf("  Phase 1: Capturing and Instantiating CUDA Graph...\n");
    gpuCheck(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal), "begin capture");
    
    // Capture H2D transfers
    nvtxRangePush("Single_GPU_Graph_Capture_H2D");
    gpuCheck(cudaMemcpyAsync(dev_X_single, batch_X, batchBytes, cudaMemcpyHostToDevice, captureStream), "capture copy X");
    gpuCheck(cudaMemcpyAsync(dev_H_single, batch_H, batchBytes, cudaMemcpyHostToDevice, captureStream), "capture copy H");
    nvtxRangePop(); // Single_GPU_Graph_Capture_H2D
    
    // Capture kernel launches
    nvtxRangePush("Single_GPU_Graph_Capture_Kernels");
    dim3 block(BLOCKSIZE);
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
        int symbolsPerChunk = BATCH_SIZE_SYMBOLS / STREAMS_PER_GPU;
        int remainingSymbols = BATCH_SIZE_SYMBOLS % STREAMS_PER_GPU;
        int chunkStartSymbol = s * symbolsPerChunk;
        int chunkSymbols = symbolsPerChunk + (s == STREAMS_PER_GPU - 1 ? remainingSymbols : 0);
        if (chunkSymbols > 0) {
            int chunkElements = chunkSymbols * ELEMENTS_PER_SYMBOL;
            size_t chunkOffset = (size_t)chunkStartSymbol * ELEMENTS_PER_SYMBOL;
            dim3 grid((chunkElements + BLOCKSIZE - 1) / BLOCKSIZE);
            pw_multiply_half_kernel<<<grid, block, 0, captureStream>>>(
                (ComplexType*)((char*)dev_X_single + chunkOffset), (ComplexType*)((char*)dev_H_single + chunkOffset),
                (ComplexType*)((char*)dev_Y_single + chunkOffset), chunkElements);
            captured_kernels++;
        }
    }
    gpuCheck(cudaGetLastError(), "kernel launch capture");
    nvtxRangePop(); // Single_GPU_Graph_Capture_Kernels
    
    // Capture D2H transfer
    nvtxRangePush("Single_GPU_Graph_Capture_D2H");
    gpuCheck(cudaMemcpyAsync(batch_Y, dev_Y_single, batchBytes, cudaMemcpyDeviceToHost, captureStream), "capture copy Y");
    nvtxRangePop(); // Single_GPU_Graph_Capture_D2H
    
    gpuCheck(cudaStreamEndCapture(captureStream, &graph), "end capture");
    gpuCheck(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0), "instantiate graph");
    printf("  Graph setup complete for %d kernels.\n", captured_kernels);
    nvtxRangePop(); // Single_GPU_CUDA_Graph_Creation

    nvtxRangePush("Single_GPU_Execution");
    printf("  Phase 2: Launching Graph and Synchronizing...\n");
    cudaEvent_t start_event, stop_event;
    gpuCheck(cudaEventCreate(&start_event), "create start");
    gpuCheck(cudaEventCreate(&stop_event), "create stop");

    // Record start time BEFORE launching the graph
    gpuCheck(cudaEventRecord(start_event, captureStream), "record start in stream");
    
    // To measure CPU launch overhead, we time the launch call itself
    float launch_overhead_ms;
    cudaEvent_t launch_start, launch_stop;
    gpuCheck(cudaEventCreate(&launch_start), "create launch start");
    gpuCheck(cudaEventCreate(&launch_stop), "create launch stop");
    
    // Measure launch overhead
    gpuCheck(cudaEventRecord(launch_start), "record launch start");
    nvtxRangePush("Single_GPU_Graph_Launch");
    gpuCheck(cudaGraphLaunch(graphExec, captureStream), "launch graph");
    nvtxRangePop(); // Single_GPU_Graph_Launch
    gpuCheck(cudaEventRecord(launch_stop), "record launch stop");
    
    // Record end time AFTER the graph completes
    gpuCheck(cudaEventRecord(stop_event, captureStream), "record stop in stream");
    
    // Wait for everything to finish
    gpuCheck(cudaEventSynchronize(stop_event), "sync stop event");
    gpuCheck(cudaEventSynchronize(launch_stop), "sync launch stop event");
    
    // Calculate timings
    gpuCheck(cudaEventElapsedTime(&launch_overhead_ms, launch_start, launch_stop), "get launch time");
    gpuCheck(cudaEventElapsedTime(&singleGpuResults.total_time, start_event, stop_event), "get total gpu time");
    nvtxRangePop(); // Single_GPU_Execution

    // Populate results
    singleGpuResults.kernel_launch_time = launch_overhead_ms;
    singleGpuResults.kernel_exec_time = singleGpuResults.total_time - launch_overhead_ms; // Execution time is total minus launch
    singleGpuResults.num_kernels_launched = 1;

    printf("  Single GPU Results: Total=%.2fms, Launch=%.2fms, Exec=%.2fms\n", 
           singleGpuResults.total_time, singleGpuResults.kernel_launch_time, singleGpuResults.kernel_exec_time);

    gpuCheck(cudaEventDestroy(start_event), "destroy start");
    gpuCheck(cudaEventDestroy(stop_event), "destroy stop");
    gpuCheck(cudaEventDestroy(launch_start), "destroy launch start");
    gpuCheck(cudaEventDestroy(launch_stop), "destroy launch stop");

    nvtxRangePop(); // Single_GPU_Performance

    // ==============================================
    // MULTI-GPU PERFORMANCE (Same Dataset Distributed)
    // ==============================================
    nvtxRangePush("Multi_GPU_Performance");
    printf("\n=== MULTI-GPU PERFORMANCE (Same Dataset Distributed) ===\n");
    
    nvtxRangePush("Multi_GPU_Memory_Allocation");
    ComplexType **dev_X = new ComplexType*[deviceCount];
    ComplexType **dev_H = new ComplexType*[deviceCount];
    ComplexType **dev_Y = new ComplexType*[deviceCount];
    cudaStream_t **streams = new cudaStream_t*[deviceCount];
    
    printf("Allocating GPU memory across %d devices...\n", deviceCount);
    for (int dev = 0; dev < deviceCount; dev++) {
        nvtxRangePushA(("GPU" + std::to_string(dev) + "_Memory_Allocation").c_str());
        gpuCheck(cudaSetDevice(dev), "set device");
        int base = BATCH_SIZE_SYMBOLS / deviceCount;
        int remainder = BATCH_SIZE_SYMBOLS % deviceCount;
        int gpuSymbols = base + (dev < remainder ? 1 : 0);
        size_t gpuBytes = (size_t)gpuSymbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
        gpuCheck(cudaMalloc(&dev_X[dev], gpuBytes), "malloc X");
        gpuCheck(cudaMalloc(&dev_H[dev], gpuBytes), "malloc H");
        gpuCheck(cudaMalloc(&dev_Y[dev], gpuBytes), "malloc Y");
        streams[dev] = new cudaStream_t[STREAMS_PER_GPU];
        for (int s = 0; s < STREAMS_PER_GPU; s++) {
            gpuCheck(cudaStreamCreate(&streams[dev][s]), "create stream");
        }
        nvtxRangePop(); // GPUx_Memory_Allocation
    }
    nvtxRangePop(); // Multi_GPU_Memory_Allocation
    
    cudaEvent_t multi_start, multi_stop;
    gpuCheck(cudaEventCreate(&multi_start), "create multi start");
    gpuCheck(cudaEventCreate(&multi_stop), "create multi stop");
    
    TimingResults multiGpuResults = {0};
    
    gpuCheck(cudaEventRecord(multi_start), "record multi start");
    
    // H2D Transfer Phase
    nvtxRangePush("Multi_GPU_H2D_Transfer");
    cudaEvent_t h2d_start, h2d_stop;
    gpuCheck(cudaEventCreate(&h2d_start), "create h2d start");
    gpuCheck(cudaEventCreate(&h2d_stop), "create h2d stop");
    gpuCheck(cudaEventRecord(h2d_start), "record h2d start");
    
    size_t currentHostOffsetSymbols = 0;
    int base = BATCH_SIZE_SYMBOLS / deviceCount;
    int remainder = BATCH_SIZE_SYMBOLS % deviceCount;
    for (int dev = 0; dev < deviceCount; dev++) {
        nvtxRangePushA(("GPU" + std::to_string(dev) + "_H2D_Transfer").c_str());
        gpuCheck(cudaSetDevice(dev), "set device");
        int gpuSymbols = base + (dev < remainder ? 1 : 0);
        size_t gpuBytes = (size_t)gpuSymbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
        gpuCheck(cudaMemcpyAsync(dev_X[dev], &batch_X[currentHostOffsetSymbols * ELEMENTS_PER_SYMBOL], gpuBytes, cudaMemcpyHostToDevice, streams[dev][0]), "H2D X");
        gpuCheck(cudaMemcpyAsync(dev_H[dev], &batch_H[currentHostOffsetSymbols * ELEMENTS_PER_SYMBOL], gpuBytes, cudaMemcpyHostToDevice, streams[dev][0]), "H2D H");
        currentHostOffsetSymbols += gpuSymbols;
        nvtxRangePop(); // GPUx_H2D_Transfer
    }
    for (int dev = 0; dev < deviceCount; dev++) gpuCheck(cudaStreamSynchronize(streams[dev][0]), "sync h2d stream");
    gpuCheck(cudaEventRecord(h2d_stop), "record h2d stop");
    nvtxRangePop(); // Multi_GPU_H2D_Transfer
    
    // Kernel Execution Phase
    nvtxRangePush("Multi_GPU_Kernel_Execution");
    cudaEvent_t exec_start, exec_stop;
    gpuCheck(cudaEventCreate(&exec_start), "create exec start");
    gpuCheck(cudaEventCreate(&exec_stop), "create exec stop");
    gpuCheck(cudaEventRecord(exec_start), "record exec start");

    int totalKernelsLaunched = 0;
    for (int dev = 0; dev < deviceCount; dev++) {
        nvtxRangePushA(("GPU" + std::to_string(dev) + "_Kernel_Launch").c_str());
        gpuCheck(cudaSetDevice(dev), "set device");
        int gpuSymbols = base + (dev < remainder ? 1 : 0);
        int symbolsPerStream = (gpuSymbols + STREAMS_PER_GPU - 1) / STREAMS_PER_GPU;
        for (int s = 0; s < STREAMS_PER_GPU && s * symbolsPerStream < gpuSymbols; s++) {
            int streamStartSymbol = s * symbolsPerStream;
            int streamSymbols = std::min(symbolsPerStream, gpuSymbols - streamStartSymbol);
            if (streamSymbols <= 0) continue;
            int streamElements = streamSymbols * ELEMENTS_PER_SYMBOL;
            size_t streamOffset = (size_t)streamStartSymbol * ELEMENTS_PER_SYMBOL;
            dim3 streamGrid((streamElements + BLOCKSIZE - 1) / BLOCKSIZE);
            pw_multiply_half_kernel<<<streamGrid, block, 0, streams[dev][s]>>>(
                (ComplexType*)((char*)dev_X[dev] + streamOffset), (ComplexType*)((char*)dev_H[dev] + streamOffset),
                (ComplexType*)((char*)dev_Y[dev] + streamOffset), streamElements);
            totalKernelsLaunched++;
        }
        nvtxRangePop(); // GPUx_Kernel_Launch
    }
    
    nvtxRangePush("Multi_GPU_Kernel_Synchronization");
    for (int dev = 0; dev < deviceCount; dev++) for (int s = 0; s < STREAMS_PER_GPU; s++) gpuCheck(cudaStreamSynchronize(streams[dev][s]), "sync kernel stream");
    nvtxRangePop(); // Multi_GPU_Kernel_Synchronization
    
    gpuCheck(cudaEventRecord(exec_stop), "record exec stop");
    nvtxRangePop(); // Multi_GPU_Kernel_Execution

    // D2H Transfer Phase
    nvtxRangePush("Multi_GPU_D2H_Transfer");
    cudaEvent_t d2h_start, d2h_stop;
    gpuCheck(cudaEventCreate(&d2h_start), "create d2h start");
    gpuCheck(cudaEventCreate(&d2h_stop), "create d2h stop");
    gpuCheck(cudaEventRecord(d2h_start), "record d2h start");
    currentHostOffsetSymbols = 0;
    for (int dev = 0; dev < deviceCount; dev++) {
        nvtxRangePushA(("GPU" + std::to_string(dev) + "_D2H_Transfer").c_str());
        gpuCheck(cudaSetDevice(dev), "set device");
        int gpuSymbols = base + (dev < remainder ? 1 : 0);
        size_t gpuBytes = (size_t)gpuSymbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
        gpuCheck(cudaMemcpyAsync(&batch_Y[currentHostOffsetSymbols * ELEMENTS_PER_SYMBOL], dev_Y[dev], gpuBytes, cudaMemcpyDeviceToHost, streams[dev][0]), "D2H Y");
        currentHostOffsetSymbols += gpuSymbols;
        nvtxRangePop(); // GPUx_D2H_Transfer
    }
    for (int dev = 0; dev < deviceCount; dev++) gpuCheck(cudaStreamSynchronize(streams[dev][0]), "sync d2h stream");
    gpuCheck(cudaEventRecord(d2h_stop), "record d2h stop");
    nvtxRangePop(); // Multi_GPU_D2H_Transfer
    
    gpuCheck(cudaEventRecord(multi_stop), "record multi stop");
    gpuCheck(cudaEventSynchronize(multi_stop), "sync multi stop");

    gpuCheck(cudaEventElapsedTime(&multiGpuResults.h2d_time, h2d_start, h2d_stop), "elapsed h2d");
    gpuCheck(cudaEventElapsedTime(&multiGpuResults.kernel_exec_time, exec_start, exec_stop), "elapsed exec");
    gpuCheck(cudaEventElapsedTime(&multiGpuResults.d2h_time, d2h_start, d2h_stop), "elapsed d2h");
    gpuCheck(cudaEventElapsedTime(&multiGpuResults.total_time, multi_start, multi_stop), "elapsed total");
    multiGpuResults.num_kernels_launched = totalKernelsLaunched;

    nvtxRangePop(); // Multi_GPU_Performance

    // ==============================================
    // DETAILED PERFORMANCE COMPARISON
    // ==============================================
    nvtxRangePush("Performance_Analysis");
    printf("\n=== DETAILED PERFORMANCE COMPARISON ===\n");
    printf("\n📊 DETAILED TIMING BREAKDOWN:\n");
    printf("                          Single GPU    Multi-GPU (%d)    Speedup\n", deviceCount);
    printf("                          -----------    --------------    -------\n");
    printf("H2D Transfer:           %8s      %8.2f ms      %s\n", "N/A*", multiGpuResults.h2d_time, "N/A");
    
    float multiLaunchTime = multiGpuResults.total_time - multiGpuResults.h2d_time - multiGpuResults.kernel_exec_time - multiGpuResults.d2h_time;
    float h2dSpeedup = 0.0f; // Can't calculate since single GPU H2D is in graph
    float launchSpeedup = singleGpuResults.kernel_launch_time / multiLaunchTime;
    float execSpeedup = singleGpuResults.kernel_exec_time / multiGpuResults.kernel_exec_time;
    float d2hSpeedup = 0.0f; // Can't calculate since single GPU D2H is in graph
    float totalSpeedup = singleGpuResults.total_time / multiGpuResults.total_time;
    
    printf("Kernel/Graph Launch:    %8.3f ms      %8.2f ms      %6.2fx\n", 
           singleGpuResults.kernel_launch_time, multiLaunchTime, launchSpeedup);
    printf("Kernel/Graph Execution: %8.2f ms      %8.2f ms      %6.2fx\n", 
           singleGpuResults.kernel_exec_time, multiGpuResults.kernel_exec_time, execSpeedup);
    printf("D2H Transfer:           %8s      %8.2f ms      %s\n", "N/A*", multiGpuResults.d2h_time, "N/A");
    printf("---------------------     -----------    --------------    -------\n");
    printf("TOTAL TIME:             %8.2f ms      %8.2f ms      %6.2fx\n", 
           singleGpuResults.total_time, multiGpuResults.total_time, totalSpeedup);
    printf("*N/A: For graphs, transfers are part of total execution time.\n");
    
    printf("\n📈 PERFORMANCE METRICS:\n");
    printf("Single GPU (with CUDA Graph):\n");
    printf("  Graph launches:         %d (from %d kernels)\n", singleGpuResults.num_kernels_launched, captured_kernels);
    printf("  Launch overhead/graph:  %.2f μs\n", singleGpuResults.kernel_launch_time * 1000.0f);
    
    if (singleGpuResults.total_time > 0.0f) {
        printf("  Complex ops/second:     %.1f billion ops/s\n", (float)totalOperations / (singleGpuResults.total_time / 1000.0f) / 1000000000.0f);
    } else {
        printf("  Complex ops/second:     ERROR - Invalid timing\n");
    }
    
    printf("\nMulti-GPU (%d devices):\n", deviceCount);
    printf("  Kernels launched:       %d\n", multiGpuResults.num_kernels_launched);
    printf("  Complex ops/second:     %.1f billion ops/s\n", (float)totalOperations / (multiGpuResults.total_time / 1000.0f) / 1000000000.0f);
    
    printf("\n🚀 EFFICIENCY ANALYSIS:\n");
    if (singleGpuResults.total_time > 0.0f && multiGpuResults.total_time > 0.0f) {
        printf("Total Time Speedup: %.2fx\n", totalSpeedup);
        printf("Parallel Efficiency: %.1f%% (%.1f%% is ideal)\n", 
               (totalSpeedup / deviceCount) * 100.0f, 100.0f);
        
        if (totalSpeedup > 2.0f) {
            printf("✅ EXCELLENT multi-GPU scaling!\n");
        } else if (totalSpeedup > 1.5f) {
            printf("✅ GOOD multi-GPU scaling\n");
        } else {
            printf("⚠️ LIMITED multi-GPU scaling\n");
        }
    } else {
        printf("❌ ERROR: Invalid timing measurements\n");
        printf("Single GPU total: %.3f ms\n", singleGpuResults.total_time);
        printf("Multi-GPU total: %.3f ms\n", multiGpuResults.total_time);
    }
    nvtxRangePop(); // Performance_Analysis
    
    // Cleanup
    nvtxRangePush("Cleanup");
    printf("\nCleaning up memory...\n");
    safeFree(batch_X); safeFree(batch_H); safeFree(batch_Y);
    gpuCheck(cudaSetDevice(0), "set device 0");
    gpuCheck(cudaFree(dev_X_single), "free X single");
    gpuCheck(cudaFree(dev_H_single), "free H single");
    gpuCheck(cudaFree(dev_Y_single), "free Y single");
    if (graphExec != nullptr) gpuCheck(cudaGraphExecDestroy(graphExec), "destroy graph exec");
    if (graph != nullptr) gpuCheck(cudaGraphDestroy(graph), "destroy graph");
    gpuCheck(cudaStreamDestroy(captureStream), "destroy capture stream");
    for (int dev = 0; dev < deviceCount; dev++) {
        gpuCheck(cudaSetDevice(dev), "set device");
        for (int s = 0; s < STREAMS_PER_GPU; s++) gpuCheck(cudaStreamDestroy(streams[dev][s]), "destroy stream");
        delete[] streams[dev];
        gpuCheck(cudaFree(dev_X[dev]), "free X");
        gpuCheck(cudaFree(dev_H[dev]), "free H");
        gpuCheck(cudaFree(dev_Y[dev]), "free Y");
    }
    delete[] dev_X; delete[] dev_H; delete[] dev_Y; delete[] streams;
    nvtxRangePop(); // Cleanup
    
    printf("\n1K rays processing and analysis complete!\n");
    return 0;
}