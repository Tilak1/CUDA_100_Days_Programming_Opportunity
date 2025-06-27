#include <cstdio>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <random>
#include <vector>
#include <string>
#include <algorithm> // For std::min
#include <nvtx3/nvToolsExt.h>  // NVTX for profiling markers

constexpr int ELEMENTS_PER_SYMBOL = 4096;
constexpr int SYMBOLS_PER_RAY = 20;
constexpr int TOTAL_RAYS = 1000;
constexpr int BLOCKSIZE = 128;
constexpr int STREAMS_PER_GPU = 64;

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
    int deviceCount;
    gpuCheck(cudaGetDeviceCount(&deviceCount), "get device count");
    if (deviceCount < 1) { printf("No CUDA devices found!\n"); return 1; }
    printf("=== Multi-GPU Point-wise Complex Multiplication Analysis ===\n");
    printf("Found %d CUDA device(s)\n", deviceCount);
    long long totalOperations = (long long)TOTAL_RAYS * SYMBOLS_PER_RAY * ELEMENTS_PER_SYMBOL;
    printf("Workload: %lld operations (%.1f million)\n", totalOperations, totalOperations / 1000000.0f);

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
    // SINGLE GPU BASELINE: DETAILED PHASE TIMING
    // ==============================================
    TimingResults singleGpuResults = {0};
    {
        printf("\n=== SINGLE GPU PERFORMANCE (Detailed Phase Analysis) ===\n");
        
        ComplexType *d_X, *d_H, *d_Y;
        gpuCheck(cudaSetDevice(0), "set device 0");
        gpuCheck(cudaMalloc(&d_X, batchBytes), "malloc X single");
        gpuCheck(cudaMalloc(&d_H, batchBytes), "malloc H single");
        gpuCheck(cudaMalloc(&d_Y, batchBytes), "malloc Y single");
        
        cudaStream_t stream;
        cudaEvent_t start, stop;
        gpuCheck(cudaStreamCreate(&stream), "create stream");
        gpuCheck(cudaEventCreate(&start), "create start");
        gpuCheck(cudaEventCreate(&stop), "create stop");
        
        // --- Phase 1: Time H2D ---
        nvtxRangePush("Single_GPU_H2D");
        gpuCheck(cudaEventRecord(start, stream), "record h2d start");
        gpuCheck(cudaMemcpyAsync(d_X, batch_X, batchBytes, cudaMemcpyHostToDevice, stream), "H2D X");
        gpuCheck(cudaMemcpyAsync(d_H, batch_H, batchBytes, cudaMemcpyHostToDevice, stream), "H2D H");
        gpuCheck(cudaEventRecord(stop, stream), "record h2d stop");
        gpuCheck(cudaEventSynchronize(stop), "sync h2d");
        gpuCheck(cudaEventElapsedTime(&singleGpuResults.h2d_time, start, stop), "get h2d time");

        // --- Phase 2: Time Kernel Graph ---
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        gpuCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin kernel capture");
        dim3 block(BLOCKSIZE);
        int captured_kernels = 0;
        int symbolsPerKernel = BATCH_SIZE_SYMBOLS / STREAMS_PER_GPU;
        for (int s = 0; s < STREAMS_PER_GPU; s++) {
            int chunkStartSymbol = s * symbolsPerKernel;
            int chunkElements = symbolsPerKernel * ELEMENTS_PER_SYMBOL;
            if (s == STREAMS_PER_GPU - 1) chunkElements = (BATCH_SIZE_SYMBOLS - chunkStartSymbol) * ELEMENTS_PER_SYMBOL;
            if (chunkElements <= 0) continue;
            size_t offset = (size_t)chunkStartSymbol * ELEMENTS_PER_SYMBOL;
            dim3 grid((chunkElements + BLOCKSIZE - 1) / BLOCKSIZE);
            pw_multiply_half_kernel<<<grid, block, 0, stream>>>(
                (ComplexType*)((char*)d_X + offset), (ComplexType*)((char*)d_H + offset),
                (ComplexType*)((char*)d_Y + offset), chunkElements);
            captured_kernels++;
        }
        gpuCheck(cudaStreamEndCapture(stream, &graph), "end kernel capture");
        gpuCheck(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0), "instantiate kernel graph");
        singleGpuResults.num_kernels_launched = captured_kernels;

        // Time the CPU launch overhead
        cudaEvent_t launch_start_cpu, launch_stop_cpu;
        gpuCheck(cudaEventCreateWithFlags(&launch_start_cpu, cudaEventBlockingSync), "create launch start cpu");
        gpuCheck(cudaEventCreateWithFlags(&launch_stop_cpu, cudaEventBlockingSync), "create launch stop cpu");
        gpuCheck(cudaEventRecord(launch_start_cpu), "record launch start cpu");
        gpuCheck(cudaGraphLaunch(graphExec, stream), "launch graph for timing");
        gpuCheck(cudaEventRecord(launch_stop_cpu), "record launch stop cpu");
        gpuCheck(cudaEventSynchronize(launch_stop_cpu), "sync launch stop cpu");
        gpuCheck(cudaEventElapsedTime(&singleGpuResults.kernel_launch_time, launch_start_cpu, launch_stop_cpu), "get launch time");
        gpuCheck(cudaEventDestroy(launch_start_cpu), "destroy launch start cpu");
        gpuCheck(cudaEventDestroy(launch_stop_cpu), "destroy launch stop cpu");

        // Time the GPU execution of the graph
        gpuCheck(cudaEventRecord(start, stream), "record exec start");
        gpuCheck(cudaGraphLaunch(graphExec, stream), "launch exec graph");
        gpuCheck(cudaEventRecord(stop, stream), "record exec stop");
        gpuCheck(cudaEventSynchronize(stop), "sync exec");
        gpuCheck(cudaEventElapsedTime(&singleGpuResults.kernel_exec_time, start, stop), "get exec time");

        // --- Phase 3: Time D2H ---
        gpuCheck(cudaEventRecord(start, stream), "record d2h start");
        gpuCheck(cudaMemcpyAsync(batch_Y, d_Y, batchBytes, cudaMemcpyDeviceToHost, stream), "D2H Y");
        gpuCheck(cudaEventRecord(stop, stream), "record d2h stop");
        gpuCheck(cudaEventSynchronize(stop), "sync d2h");
        gpuCheck(cudaEventElapsedTime(&singleGpuResults.d2h_time, start, stop), "get d2h time");
        
        singleGpuResults.total_time = singleGpuResults.h2d_time + singleGpuResults.kernel_launch_time + singleGpuResults.kernel_exec_time + singleGpuResults.d2h_time;

        gpuCheck(cudaGraphExecDestroy(graphExec), "destroy exec"); gpuCheck(cudaGraphDestroy(graph), "destroy graph");
        gpuCheck(cudaFree(d_X), "free X single"); gpuCheck(cudaFree(d_H), "free H single"); gpuCheck(cudaFree(d_Y), "free Y single");
        gpuCheck(cudaStreamDestroy(stream), "destroy stream"); gpuCheck(cudaEventDestroy(start), "destroy start"); gpuCheck(cudaEventDestroy(stop), "destroy stop");
    }
    
    // ==============================================
    // MULTI-GPU PERFORMANCE: DETAILED PHASE TIMING
    // ==============================================
    TimingResults multiGpuResults = {0};
    {
        printf("\n=== MULTI-GPU PERFORMANCE (Detailed Phase Analysis) ===\n");
        
        ComplexType **dev_X = new ComplexType*[deviceCount];
        ComplexType **dev_H = new ComplexType*[deviceCount];
        ComplexType **dev_Y = new ComplexType*[deviceCount];
        cudaStream_t **streams = new cudaStream_t*[deviceCount];
        
        int base = BATCH_SIZE_SYMBOLS / deviceCount;
        int remainder = BATCH_SIZE_SYMBOLS % deviceCount;
        for (int dev = 0; dev < deviceCount; dev++) {
            gpuCheck(cudaSetDevice(dev), "set device multi");
            int gpuSymbols = base + (dev < remainder ? 1 : 0);
            size_t gpuBytes = (size_t)gpuSymbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
            gpuCheck(cudaMalloc(&dev_X[dev], gpuBytes), "malloc X multi"); 
            gpuCheck(cudaMalloc(&dev_H[dev], gpuBytes), "malloc H multi"); 
            gpuCheck(cudaMalloc(&dev_Y[dev], gpuBytes), "malloc Y multi");
            streams[dev] = new cudaStream_t[STREAMS_PER_GPU];
            for (int s = 0; s < STREAMS_PER_GPU; s++) gpuCheck(cudaStreamCreate(&streams[dev][s]), "create stream multi");
        }
        
        cudaEvent_t start, stop;
        gpuCheck(cudaEventCreate(&start), "create start multi"); 
        gpuCheck(cudaEventCreate(&stop), "create stop multi");
        
        // Time H2D
        gpuCheck(cudaEventRecord(start, 0), "record multi h2d start");
        size_t host_offset_symbols = 0;
        for (int dev = 0; dev < deviceCount; dev++) {
            gpuCheck(cudaSetDevice(dev), "set device multi h2d");
            int gpuSymbols = base + (dev < remainder ? 1 : 0);
            size_t gpuBytes = (size_t)gpuSymbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
            gpuCheck(cudaMemcpyAsync(dev_X[dev], batch_X + host_offset_symbols, gpuBytes, cudaMemcpyHostToDevice, streams[dev][0]), "multi h2d X");
            gpuCheck(cudaMemcpyAsync(dev_H[dev], batch_H + host_offset_symbols, gpuBytes, cudaMemcpyHostToDevice, streams[dev][0]), "multi h2d H");
            host_offset_symbols += gpuSymbols;
        }
        for (int dev=0; dev<deviceCount; ++dev) { gpuCheck(cudaSetDevice(dev), "set dev sync h2d"); gpuCheck(cudaStreamSynchronize(streams[dev][0]), "sync h2d multi"); }
        gpuCheck(cudaEventRecord(stop, 0), "record multi h2d stop");
        gpuCheck(cudaEventSynchronize(stop), "sync h2d multi");
        gpuCheck(cudaEventElapsedTime(&multiGpuResults.h2d_time, start, stop), "get h2d time multi");
        
        // Time Kernel Launch + Exec
        cudaEvent_t launch_start_cpu, launch_stop_cpu;
        gpuCheck(cudaEventCreateWithFlags(&launch_start_cpu, cudaEventBlockingSync), "create launch start cpu multi");
        gpuCheck(cudaEventCreateWithFlags(&launch_stop_cpu, cudaEventBlockingSync), "create launch stop cpu multi");
        
        gpuCheck(cudaEventRecord(start, 0), "record multi exec start");
        gpuCheck(cudaEventRecord(launch_start_cpu, 0), "record multi launch start");
        int totalKernels = 0;
        dim3 block(BLOCKSIZE); // <--- ADD THIS LINE BACK


        for (int dev=0; dev<deviceCount; ++dev) {
            gpuCheck(cudaSetDevice(dev), "set dev multi exec");
            int gpuSymbols = base + (dev < remainder ? 1 : 0);
            int symbolsPerStream = (gpuSymbols + STREAMS_PER_GPU - 1) / STREAMS_PER_GPU;
            for (int s = 0; s < STREAMS_PER_GPU && s * symbolsPerStream < gpuSymbols; s++) {
                int streamStartSymbol = s * symbolsPerStream;
                int streamSymbols = std::min(symbolsPerStream, gpuSymbols - streamStartSymbol);
                if (streamSymbols <= 0) continue;
                int streamElements = streamSymbols * ELEMENTS_PER_SYMBOL;
                size_t streamOffset = (size_t)streamStartSymbol * ELEMENTS_PER_SYMBOL;
                dim3 grid((streamElements + BLOCKSIZE - 1) / BLOCKSIZE);
                pw_multiply_half_kernel<<<grid, block, 0, streams[dev][s]>>>(
                    (ComplexType*)((char*)dev_X[dev] + streamOffset), (ComplexType*)((char*)dev_H[dev] + streamOffset),
                    (ComplexType*)((char*)dev_Y[dev] + streamOffset), streamElements);
                totalKernels++;
            }
        }
        gpuCheck(cudaEventRecord(launch_stop_cpu, 0), "record multi launch stop");
        
        for (int dev=0; dev<deviceCount; ++dev) {
            gpuCheck(cudaSetDevice(dev), "set dev sync exec");
            for(int s=0; s<STREAMS_PER_GPU; ++s) gpuCheck(cudaStreamSynchronize(streams[dev][s]), "sync exec stream multi");
        }
        gpuCheck(cudaEventRecord(stop, 0), "record multi exec stop");
        gpuCheck(cudaEventSynchronize(stop), "sync exec multi");
        gpuCheck(cudaEventSynchronize(launch_stop_cpu), "sync launch multi");
        
        float total_exec_time;
        gpuCheck(cudaEventElapsedTime(&total_exec_time, start, stop), "get total exec time multi");
        gpuCheck(cudaEventElapsedTime(&multiGpuResults.kernel_launch_time, launch_start_cpu, launch_stop_cpu), "get launch time multi");
        multiGpuResults.kernel_exec_time = total_exec_time - multiGpuResults.kernel_launch_time; 
        multiGpuResults.num_kernels_launched = totalKernels;
        
        // Time D2H
        gpuCheck(cudaEventRecord(start, 0), "record multi d2h start");
        host_offset_symbols = 0;
        for (int dev = 0; dev < deviceCount; dev++) {
            gpuCheck(cudaSetDevice(dev), "set dev multi d2h");
            int gpuSymbols = base + (dev < remainder ? 1 : 0);
            size_t gpuBytes = (size_t)gpuSymbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
            gpuCheck(cudaMemcpyAsync(batch_Y + host_offset_symbols, dev_Y[dev], gpuBytes, cudaMemcpyDeviceToHost, streams[dev][0]), "multi d2h Y");
            host_offset_symbols += gpuSymbols;
        }
        for (int dev=0; dev<deviceCount; ++dev) { gpuCheck(cudaSetDevice(dev), "set dev sync d2h"); gpuCheck(cudaStreamSynchronize(streams[dev][0]), "sync d2h multi"); }
        gpuCheck(cudaEventRecord(stop, 0), "record multi d2h stop");
        gpuCheck(cudaEventSynchronize(stop), "sync d2h multi");
        gpuCheck(cudaEventElapsedTime(&multiGpuResults.d2h_time, start, stop), "get d2h time multi");

        multiGpuResults.total_time = multiGpuResults.h2d_time + multiGpuResults.kernel_launch_time + multiGpuResults.kernel_exec_time + multiGpuResults.d2h_time;

        for(int dev=0; dev<deviceCount; ++dev) {
            gpuCheck(cudaSetDevice(dev), "set dev cleanup");
            for(int s=0; s<STREAMS_PER_GPU; ++s) gpuCheck(cudaStreamDestroy(streams[dev][s]), "destroy stream multi");
            delete[] streams[dev];
            gpuCheck(cudaFree(dev_X[dev]), "free X multi"); gpuCheck(cudaFree(dev_H[dev]), "free H multi"); gpuCheck(cudaFree(dev_Y[dev]), "free Y multi");
        }
        delete[] dev_X; delete[] dev_H; delete[] dev_Y; delete[] streams;
        gpuCheck(cudaEventDestroy(start), "destroy start multi"); gpuCheck(cudaEventDestroy(stop), "destroy stop multi");
        gpuCheck(cudaEventDestroy(launch_start_cpu), "destroy launch start cpu multi"); gpuCheck(cudaEventDestroy(launch_stop_cpu), "destroy launch stop cpu multi");
    }
    
    // ==============================================
    // DETAILED PERFORMANCE COMPARISON
    // ==============================================
    printf("\n=== DETAILED PERFORMANCE COMPARISON ===\n");
    printf("\n📊 DETAILED TIMING BREAKDOWN:\n");
    printf("                          Single GPU    Multi-GPU (%d)    Speedup\n", deviceCount);
    printf("                          -----------    --------------    -------\n");
    printf("H2D Transfer:           %8.2f ms      %8.2f ms      %6.2fx\n", singleGpuResults.h2d_time, multiGpuResults.h2d_time, singleGpuResults.h2d_time / multiGpuResults.h2d_time);
    printf("Kernel Launch:          %8.3f ms      %8.3f ms      %6.2fx\n", singleGpuResults.kernel_launch_time, multiGpuResults.kernel_launch_time, singleGpuResults.kernel_launch_time / multiGpuResults.kernel_launch_time);
    printf("Kernel Execution:       %8.2f ms      %8.2f ms      %6.2fx\n", singleGpuResults.kernel_exec_time, multiGpuResults.kernel_exec_time, singleGpuResults.kernel_exec_time / multiGpuResults.kernel_exec_time);
    printf("D2H Transfer:           %8.2f ms      %8.2f ms      %6.2fx\n", singleGpuResults.d2h_time, multiGpuResults.d2h_time, singleGpuResults.d2h_time / multiGpuResults.d2h_time);
    printf("---------------------     -----------    --------------    -------\n");
    printf("TOTAL TIME:             %8.2f ms      %8.2f ms      %6.2fx\n", singleGpuResults.total_time, multiGpuResults.total_time, singleGpuResults.total_time / multiGpuResults.total_time);
    
    printf("\n📈 PERFORMANCE METRICS:\n");
    printf("Single GPU (Kernel-Only Graph):\n");
    printf("  Kernels captured:       %d\n", singleGpuResults.num_kernels_launched);
    printf("  Launch overhead/graph:  %.2f μs\n", singleGpuResults.kernel_launch_time * 1000.0f);
    
    printf("\nMulti-GPU (Standard Launch):\n");
    printf("  Kernels launched:       %d\n", multiGpuResults.num_kernels_launched);
    printf("  Launch overhead/kernel: %.2f μs\n", (multiGpuResults.kernel_launch_time / multiGpuResults.num_kernels_launched) * 1000.0f);
    
    printf("\nProcessing and analysis complete!\n");
    return 0;
}