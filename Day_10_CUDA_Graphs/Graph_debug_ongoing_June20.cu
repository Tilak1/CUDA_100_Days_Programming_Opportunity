#include <cstdio>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <random>
#include <vector>
#include <algorithm> // For std::min

constexpr int ELEMENTS_PER_SYMBOL = 4096;    // 4K elements per symbol
constexpr int SYMBOLS_PER_RAY = 20;          // 20 symbols per ray
constexpr int TOTAL_RAYS = 2000;             // 1K rays for testing
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

    // ==============================================
    // SINGLE GPU BASELINE PERFORMANCE (Corrected with CUDA Graph)
    // ==============================================
    printf("\n=== SINGLE GPU PERFORMANCE (with CUDA Graph) ===\n");
    
    ComplexType *dev_X_single, *dev_H_single, *dev_Y_single;
    gpuCheck(cudaSetDevice(0), "set device 0");
    gpuCheck(cudaMalloc(&dev_X_single, batchBytes), "malloc X single");
    gpuCheck(cudaMalloc(&dev_H_single, batchBytes), "malloc H single");
    gpuCheck(cudaMalloc(&dev_Y_single, batchBytes), "malloc Y single");
    
    cudaStream_t captureStream;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    gpuCheck(cudaStreamCreate(&captureStream), "create capture stream");
    
    TimingResults singleGpuResults = {0};
    int captured_kernels = 0;
    
    printf("  Phase 1: Capturing and Instantiating CUDA Graph...\n");
    gpuCheck(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal), "begin capture");
    gpuCheck(cudaMemcpyAsync(dev_X_single, batch_X, batchBytes, cudaMemcpyHostToDevice, captureStream), "capture copy X");
    gpuCheck(cudaMemcpyAsync(dev_H_single, batch_H, batchBytes, cudaMemcpyHostToDevice, captureStream), "capture copy H");
    dim3 block(BLOCKSIZE);
    int symbolsPerChunk = BATCH_SIZE_SYMBOLS / STREAMS_PER_GPU;
    int remainingSymbols = BATCH_SIZE_SYMBOLS % STREAMS_PER_GPU;
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
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

    printf("  Phase 2: Launching Graph and Synchronizing...\n");
    cudaEvent_t start_event, stop_event;
    gpuCheck(cudaEventCreate(&start_event), "create start");
    gpuCheck(cudaEventCreate(&stop_event), "create stop");

    // CORRECTED TIMING: Record start, launch graph, record stop.
    gpuCheck(cudaEventRecord(start_event, captureStream), "record start in stream");
    gpuCheck(cudaGraphLaunch(graphExec, captureStream), "launch graph");
    gpuCheck(cudaEventRecord(stop_event, captureStream), "record stop in stream");
    
    // Wait for the final event to ensure the graph has finished.
    gpuCheck(cudaEventSynchronize(stop_event), "sync stop event");

    // Populate results
    gpuCheck(cudaEventElapsedTime(&singleGpuResults.total_time, start_event, stop_event), "get total gpu time");
    singleGpuResults.kernel_exec_time = singleGpuResults.total_time; // GPU exec time is the total time between start and stop events in the stream
    singleGpuResults.num_kernels_launched = 1;

    // A separate, quick measurement for launch overhead (optional but good practice)
    cudaEvent_t launch_start_cpu, launch_stop_cpu;
    gpuCheck(cudaEventCreateWithFlags(&launch_start_cpu, cudaEventBlockingSync), "create launch start");
    gpuCheck(cudaEventCreateWithFlags(&launch_stop_cpu, cudaEventBlockingSync), "create launch stop");
    gpuCheck(cudaEventRecord(launch_start_cpu), "record launch start");
    gpuCheck(cudaGraphLaunch(graphExec, captureStream), "launch graph for timing");
    gpuCheck(cudaEventRecord(launch_stop_cpu), "record launch stop");
    gpuCheck(cudaEventSynchronize(launch_stop_cpu), "sync launch stop event");
    gpuCheck(cudaEventElapsedTime(&singleGpuResults.kernel_launch_time, launch_start_cpu, launch_stop_cpu), "get launch time");
    gpuCheck(cudaEventDestroy(launch_start_cpu), "destroy launch start");
    gpuCheck(cudaEventDestroy(launch_stop_cpu), "destroy launch stop");

    gpuCheck(cudaEventDestroy(start_event), "destroy start");
    gpuCheck(cudaEventDestroy(stop_event), "destroy stop");

    // ==============================================
    // MULTI-GPU PERFORMANCE (Same Dataset Distributed)
    // ==============================================
    printf("\n=== MULTI-GPU PERFORMANCE (Same Dataset Distributed) ===\n");
    
    ComplexType **dev_X = new ComplexType*[deviceCount];
    ComplexType **dev_H = new ComplexType*[deviceCount];
    ComplexType **dev_Y = new ComplexType*[deviceCount];
    cudaStream_t **streams = new cudaStream_t*[deviceCount];
    
    printf("Allocating GPU memory across %d devices...\n", deviceCount);
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
    
    cudaEvent_t multi_start, multi_stop;
    gpuCheck(cudaEventCreate(&multi_start), "create multi start");
    gpuCheck(cudaEventCreate(&multi_stop), "create multi stop");
    
    TimingResults multiGpuResults = {0};
    
    gpuCheck(cudaEventRecord(multi_start, 0), "record multi start");
    
    // H2D
    size_t currentHostOffsetSymbols = 0;
    for (int dev = 0; dev < deviceCount; dev++) {
        gpuCheck(cudaSetDevice(dev), "set device");
        int gpuSymbols = base + (dev < remainder ? 1 : 0);
        size_t gpuBytes = (size_t)gpuSymbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
        gpuCheck(cudaMemcpyAsync(dev_X[dev], &batch_X[currentHostOffsetSymbols * ELEMENTS_PER_SYMBOL], gpuBytes, cudaMemcpyHostToDevice, streams[dev][0]), "H2D X");
        gpuCheck(cudaMemcpyAsync(dev_H[dev], &batch_H[currentHostOffsetSymbols * ELEMENTS_PER_SYMBOL], gpuBytes, cudaMemcpyHostToDevice, streams[dev][0]), "H2D H");
        currentHostOffsetSymbols += gpuSymbols;
    }

    // Kernels
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

    // D2H
    currentHostOffsetSymbols = 0;
    for (int dev = 0; dev < deviceCount; dev++) {
        gpuCheck(cudaSetDevice(dev), "set device");
        int gpuSymbols = base + (dev < remainder ? 1 : 0);
        size_t gpuBytes = (size_t)gpuSymbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
        gpuCheck(cudaMemcpyAsync(&batch_Y[currentHostOffsetSymbols * ELEMENTS_PER_SYMBOL], dev_Y[dev], gpuBytes, cudaMemcpyDeviceToHost, streams[dev][0]), "D2H Y");
        currentHostOffsetSymbols += gpuSymbols;
    }
    
    // Final sync and stop
    gpuCheck(cudaDeviceSynchronize(), "Sync all devices");
    gpuCheck(cudaEventRecord(multi_stop, 0), "record multi stop");
    gpuCheck(cudaEventSynchronize(multi_stop), "sync multi stop");

    gpuCheck(cudaEventElapsedTime(&multiGpuResults.total_time, multi_start, multi_stop), "elapsed total");
    multiGpuResults.num_kernels_launched = totalKernelsLaunched;

    // ==============================================
    // DETAILED PERFORMANCE COMPARISON
    // ==============================================
    printf("\n=== DETAILED PERFORMANCE COMPARISON ===\n");
    printf("\n📊 DETAILED TIMING BREAKDOWN:\n");
    printf("                          Single GPU    Multi-GPU (%d)\n", deviceCount);
    printf("                          -----------    --------------\n");
    printf("Launch Overhead:        %8.3f ms           N/A\n", singleGpuResults.kernel_launch_time);
    printf("Total Execution Time:   %8.2f ms      %8.2f ms\n", singleGpuResults.kernel_exec_time, multiGpuResults.total_time);
    
    printf("\n📈 PERFORMANCE METRICS:\n");
    printf("Single GPU (with CUDA Graph):\n");
    printf("  Graph launches:         %d (from %d kernels)\n", singleGpuResults.num_kernels_launched, captured_kernels);
    printf("  Launch overhead/graph:  %.2f μs\n", singleGpuResults.kernel_launch_time * 1000.0f);
    printf("  Complex ops/second:     %.1f billion ops/s\n", (float)totalOperations / (singleGpuResults.total_time / 1000.0f) / 1000000000.0f);
    
    printf("\nMulti-GPU (%d devices):\n");
    printf("  Kernels launched:       %d\n", multiGpuResults.num_kernels_launched);
    printf("  Complex ops/second:     %.1f billion ops/s\n", (float)totalOperations / (multiGpuResults.total_time / 1000.0f) / 1000000000.0f);
    
    printf("\n🚀 EFFICIENCY ANALYSIS (Total Time Speedup: %.2fx):\n", singleGpuResults.total_time / multiGpuResults.total_time);
    
    // Cleanup
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
    printf("\n1K rays processing and analysis complete!\n");
    return 0;
}