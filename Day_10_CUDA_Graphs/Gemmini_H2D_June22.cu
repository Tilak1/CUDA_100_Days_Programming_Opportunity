#include <cstdio>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <random>
#include <vector>
#include <string>
#include <algorithm> // For std::min
#include <nvtx3/nvToolsExt.h>  // NVTX for profiling markers

constexpr int ELEMENTS_PER_SYMBOL = 4096;    // 4K elements per symbol
constexpr int SYMBOLS_PER_RAY = 20;          // 20 symbols per ray
constexpr int TOTAL_RAYS = 2000;             // 1K rays for testing
constexpr int BLOCKSIZE = 128;               // Threads per block
constexpr int STREAMS_PER_GPU = 64;          // Balanced for performance

constexpr int BATCH_SIZE_RAYS = 1000;
constexpr int BATCH_SIZE_SYMBOLS = BATCH_SIZE_RAYS * SYMBOLS_PER_RAY;

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
    printf("Workload: %lld operations (%.1f million)\n", totalOperations, totalOperations / 1000000.0f);
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        gpuCheck(cudaGetDeviceProperties(&prop, dev), "get device properties");
        printf("\nDevice %d: %s\n", dev, prop.name);
    }
    printf("\n");
    nvtxRangePop();

    nvtxRangePush("Host_Memory_Allocation_And_Data_Generation");
    size_t batchBytes = (size_t)BATCH_SIZE_SYMBOLS * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
    printf("Allocating %.2f GB of pinned host memory...\n", (batchBytes * 3) / (1024.0f * 1024.0f * 1024.0f));
    ComplexType *batch_X = safeHostAlloc(batchBytes, "batch_X");
    ComplexType *batch_H = safeHostAlloc(batchBytes, "batch_H");
    ComplexType *batch_Y = safeHostAlloc(batchBytes, "batch_Y");
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
    // SINGLE GPU BASELINE PERFORMANCE
    // ==============================================
    nvtxRangePush("Single_GPU_Performance");
    printf("\n=== SINGLE GPU PERFORMANCE (with CUDA Graph) ===\n");
    
    nvtxRangePush("Single_GPU_Setup");
    ComplexType *dev_X_single, *dev_H_single, *dev_Y_single;
    gpuCheck(cudaSetDevice(0), "set device 0");
    gpuCheck(cudaMalloc(&dev_X_single, batchBytes), "malloc X single");
    gpuCheck(cudaMalloc(&dev_H_single, batchBytes), "malloc H single");
    gpuCheck(cudaMalloc(&dev_Y_single, batchBytes), "malloc Y single");
    
    cudaStream_t captureStream;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    gpuCheck(cudaStreamCreate(&captureStream), "create capture stream");
    
    printf("  Phase 1: Capturing and Instantiating CUDA Graph...\n");
    gpuCheck(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal), "begin capture");
    gpuCheck(cudaMemcpyAsync(dev_X_single, batch_X, batchBytes, cudaMemcpyHostToDevice, captureStream), "capture copy X");
    gpuCheck(cudaMemcpyAsync(dev_H_single, batch_H, batchBytes, cudaMemcpyHostToDevice, captureStream), "capture copy H");
    dim3 block(BLOCKSIZE);
    int captured_kernels = 0;
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
    gpuCheck(cudaMemcpyAsync(batch_Y, dev_Y_single, batchBytes, cudaMemcpyDeviceToHost, captureStream), "capture copy Y");
    gpuCheck(cudaStreamEndCapture(captureStream, &graph), "end capture");
    gpuCheck(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0), "instantiate graph");
    printf("  Graph setup complete for %d kernels.\n", captured_kernels);
    nvtxRangePop(); 

    nvtxRangePush("Single_GPU_Execution_And_Timing");
    printf("  Phase 2: Launching Graph and Synchronizing...\n");
    
    TimingResults singleGpuResults = {0};
    cudaEvent_t start_event, stop_event;
    gpuCheck(cudaEventCreate(&start_event), "create start");
    gpuCheck(cudaEventCreate(&stop_event), "create stop");

    gpuCheck(cudaEventRecord(start_event, captureStream), "record start in stream");
    gpuCheck(cudaGraphLaunch(graphExec, captureStream), "launch graph");
    gpuCheck(cudaEventRecord(stop_event, captureStream), "record stop in stream");
    gpuCheck(cudaEventSynchronize(stop_event), "sync stop event");

    gpuCheck(cudaEventElapsedTime(&singleGpuResults.total_time, start_event, stop_event), "get total gpu time");
    
    cudaEvent_t launch_start_cpu, launch_stop_cpu;
    gpuCheck(cudaEventCreateWithFlags(&launch_start_cpu, cudaEventBlockingSync), "create launch start");
    gpuCheck(cudaEventCreateWithFlags(&launch_stop_cpu, cudaEventBlockingSync), "create launch stop");
    gpuCheck(cudaEventRecord(launch_start_cpu), "record launch start");
    gpuCheck(cudaGraphLaunch(graphExec, captureStream), "launch graph for timing");
    gpuCheck(cudaEventRecord(launch_stop_cpu), "record launch stop");
    gpuCheck(cudaEventSynchronize(launch_stop_cpu), "sync launch stop event");
    gpuCheck(cudaEventElapsedTime(&singleGpuResults.kernel_launch_time, launch_start_cpu, launch_stop_cpu), "get launch time");
    
    singleGpuResults.kernel_exec_time = singleGpuResults.total_time;
    singleGpuResults.num_kernels_launched = 1;

    gpuCheck(cudaEventDestroy(start_event), "destroy start");
    gpuCheck(cudaEventDestroy(stop_event), "destroy stop");
    gpuCheck(cudaEventDestroy(launch_start_cpu), "destroy launch start");
    gpuCheck(cudaEventDestroy(launch_stop_cpu), "destroy launch stop");
    nvtxRangePop();
    nvtxRangePop();

    // ==============================================
    // MULTI-GPU PERFORMANCE
    // ==============================================
    nvtxRangePush("Multi_GPU_Performance_Analysis_Mode");
    printf("\n=== MULTI-GPU PERFORMANCE (Serialized Analysis Mode) ===\n");
    
    nvtxRangePush("Multi_GPU_Setup");
    ComplexType **dev_X = new ComplexType*[deviceCount];
    ComplexType **dev_H = new ComplexType*[deviceCount];
    ComplexType **dev_Y = new ComplexType*[deviceCount];
    cudaStream_t **streams = new cudaStream_t*[deviceCount];
    
    int base = BATCH_SIZE_SYMBOLS / deviceCount;
    int remainder = BATCH_SIZE_SYMBOLS % deviceCount;
    for (int dev = 0; dev < deviceCount; dev++) {
        gpuCheck(cudaSetDevice(dev), "set device");
        int gpuSymbols = base + (dev < remainder ? 1 : 0);
        size_t gpuBytes = (size_t)gpuSymbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
        gpuCheck(cudaMalloc(&dev_X[dev], gpuBytes), "malloc X");
        gpuCheck(cudaMalloc(&dev_H[dev], gpuBytes), "malloc H");
        gpuCheck(cudaMalloc(&dev_Y[dev], gpuBytes), "malloc Y");
        streams[dev] = new cudaStream_t[STREAMS_PER_GPU];
        for (int s = 0; s < STREAMS_PER_GPU; s++) {
            gpuCheck(cudaStreamCreate(&streams[dev][s]), "create stream");
        }
    }
    nvtxRangePop(); 
    
    TimingResults multiGpuResults = {0};
    cudaEvent_t phase_start, phase_stop;
    gpuCheck(cudaEventCreate(&phase_start), "create phase start");
    gpuCheck(cudaEventCreate(&phase_stop), "create phase stop");

    // --- Phase 1: H2D Transfers ---
    nvtxRangePush("Multi_GPU_H2D");
    gpuCheck(cudaEventRecord(phase_start, 0), "record h2d start");
    size_t currentHostOffsetSymbols = 0;
    for (int dev = 0; dev < deviceCount; dev++) {
        gpuCheck(cudaSetDevice(dev), "set device");
        int gpuSymbols = base + (dev < remainder ? 1 : 0);
        size_t gpuBytes = (size_t)gpuSymbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
        gpuCheck(cudaMemcpyAsync(dev_X[dev], &batch_X[currentHostOffsetSymbols * ELEMENTS_PER_SYMBOL], gpuBytes, cudaMemcpyHostToDevice, streams[dev][0]), "H2D X");
        gpuCheck(cudaMemcpyAsync(dev_H[dev], &batch_H[currentHostOffsetSymbols * ELEMENTS_PER_SYMBOL], gpuBytes, cudaMemcpyHostToDevice, streams[dev][0]), "H2D H");
        currentHostOffsetSymbols += gpuSymbols;
    }
    for (int dev = 0; dev < deviceCount; dev++) { gpuCheck(cudaSetDevice(dev), "set dev"); gpuCheck(cudaStreamSynchronize(streams[dev][0]), "sync h2d stream"); }
    gpuCheck(cudaEventRecord(phase_stop, 0), "record h2d stop");
    gpuCheck(cudaEventSynchronize(phase_stop), "sync phase stop");
    gpuCheck(cudaEventElapsedTime(&multiGpuResults.h2d_time, phase_start, phase_stop), "elapsed h2d");
    nvtxRangePop();

    // --- Phase 2: Kernel Execution ---
    nvtxRangePush("Multi_GPU_Execution");
    gpuCheck(cudaEventRecord(phase_start, 0), "record exec start");
    int totalKernelsLaunched = 0;
    for (int dev = 0; dev < deviceCount; dev++) {
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
    }
    for (int dev = 0; dev < deviceCount; dev++) { gpuCheck(cudaSetDevice(dev), "set dev"); for (int s = 0; s < STREAMS_PER_GPU; s++) gpuCheck(cudaStreamSynchronize(streams[dev][s]), "sync kernel stream");}
    gpuCheck(cudaEventRecord(phase_stop, 0), "record exec stop");
    gpuCheck(cudaEventSynchronize(phase_stop), "sync phase stop");
    gpuCheck(cudaEventElapsedTime(&multiGpuResults.kernel_exec_time, phase_start, phase_stop), "elapsed exec");
    multiGpuResults.num_kernels_launched = totalKernelsLaunched;
    nvtxRangePop();

    // --- Phase 3: D2H Transfers ---
    nvtxRangePush("Multi_GPU_D2H");
    gpuCheck(cudaEventRecord(phase_start, 0), "record d2h start");
    currentHostOffsetSymbols = 0;
    for (int dev = 0; dev < deviceCount; dev++) {
        gpuCheck(cudaSetDevice(dev), "set device");
        int gpuSymbols = base + (dev < remainder ? 1 : 0);
        size_t gpuBytes = (size_t)gpuSymbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
        gpuCheck(cudaMemcpyAsync(&batch_Y[currentHostOffsetSymbols * ELEMENTS_PER_SYMBOL], dev_Y[dev], gpuBytes, cudaMemcpyDeviceToHost, streams[dev][0]), "D2H Y");
        currentHostOffsetSymbols += gpuSymbols;
    }
    for (int dev = 0; dev < deviceCount; dev++) { gpuCheck(cudaSetDevice(dev), "set dev"); gpuCheck(cudaStreamSynchronize(streams[dev][0]), "sync d2h stream");}
    gpuCheck(cudaEventRecord(phase_stop, 0), "record d2h stop");
    gpuCheck(cudaEventSynchronize(phase_stop), "sync phase stop");
    gpuCheck(cudaEventElapsedTime(&multiGpuResults.d2h_time, phase_start, phase_stop), "elapsed d2h");
    nvtxRangePop();
    
    multiGpuResults.total_time = multiGpuResults.h2d_time + multiGpuResults.kernel_exec_time + multiGpuResults.d2h_time;

    gpuCheck(cudaEventDestroy(phase_start), "destroy phase start");
    gpuCheck(cudaEventDestroy(phase_stop), "destroy phase stop");
    nvtxRangePop();


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
    multiLaunchTime = std::max(0.0f, multiLaunchTime); 
    float launchSpeedup = (multiLaunchTime > 0.0001f) ? (singleGpuResults.kernel_launch_time / multiLaunchTime) : 0.0f;
    float execSpeedup = (multiGpuResults.kernel_exec_time > 0.0001f) ? (singleGpuResults.kernel_exec_time / multiGpuResults.kernel_exec_time) : 0.0f;
    float totalSpeedup = (multiGpuResults.total_time > 0.0001f) ? (singleGpuResults.total_time / multiGpuResults.total_time) : 0.0f;
    
    printf("Kernel/Graph Launch:    %8.3f ms      %8.3f ms      %6.2fx\n", singleGpuResults.kernel_launch_time, multiLaunchTime, launchSpeedup);
    printf("Kernel/Graph Execution: %8.2f ms      %8.2f ms      %6.2fx\n", singleGpuResults.kernel_exec_time, multiGpuResults.kernel_exec_time, execSpeedup);
    printf("D2H Transfer:           %8s      %8.2f ms      %s\n", "N/A*", multiGpuResults.d2h_time, "N/A");
    printf("---------------------     -----------    --------------    -------\n");
    printf("TOTAL TIME:             %8.2f ms      %8.2f ms      %6.2fx\n", singleGpuResults.total_time, multiGpuResults.total_time, totalSpeedup);
    printf("*N/A: For graphs, transfers are part of total execution time.\n");
    
    printf("\n📈 PERFORMANCE METRICS:\n");
    printf("Single GPU (with CUDA Graph):\n");
    printf("  Graph launches:         %d (from %d kernels)\n", singleGpuResults.num_kernels_launched, captured_kernels);
    printf("  Launch overhead/graph:  %.2f μs\n", singleGpuResults.kernel_launch_time * 1000.0f);
    printf("  Complex ops/second:     %.1f billion ops/s\n", (float)totalOperations / (singleGpuResults.total_time / 1000.0f) / 1000000000.0f);
    
    printf("\nMulti-GPU (%d devices):\n");
    printf("  Kernels launched:       %d\n", multiGpuResults.num_kernels_launched);
    printf("  Complex ops/second:     %.1f billion ops/s\n", (float)totalOperations / (multiGpuResults.total_time / 1000.0f) / 1000000000.0f);
    
    printf("\n🚀 EFFICIENCY ANALYSIS:\n");
    printf("Total Time Speedup: %.2fx\n", totalSpeedup);
    printf("Parallel Efficiency: %.1f%% (%.1f%% is ideal)\n", (totalSpeedup / deviceCount) * 100.0f, 100.0f);
    nvtxRangePop();
    
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
    nvtxRangePop(); 
    
    printf("\n1K rays processing and analysis complete!\n");
    return 0;
}