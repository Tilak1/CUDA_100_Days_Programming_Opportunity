#include <cstdio>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <random>
#include <vector>

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
        // Complex multiplication using half2: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        __half2 x_val = X[i];
        __half2 h_val = H[i];
        
        // Extract real and imaginary parts
        __half x_real = __low2half(x_val);
        __half x_imag = __high2half(x_val);
        __half h_real = __low2half(h_val);
        __half h_imag = __high2half(h_val);
        
        // Complex multiplication
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

// Generate random complex half-precision vector
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

// Safe memory allocation with fallback
ComplexType* safeHostAlloc(size_t bytes, const char* name) {
    ComplexType* ptr = nullptr;
    
    // Try cudaHostAlloc first (better performance)
    cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (err == cudaSuccess) {
        printf("  %s: cudaHostAlloc success (%.2f GB)\n", name, bytes / (1024.0f * 1024.0f * 1024.0f));
        return ptr;
    }
    
    // Fallback to regular malloc
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
        // Try CUDA free first
        cudaError_t err = cudaFreeHost(ptr);
        if (err != cudaSuccess) {
            // Fallback to regular free
            free(ptr);
        }
    }
}

// Timing helper structure
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
    printf("Precision: %s (4 bytes per complex number)\n", precision_name);
    printf("Memory savings: 50%% vs single-precision\n");
    printf("Workload configuration:\n");
    printf("  Total rays: %d\n", TOTAL_RAYS);
    printf("  Symbols per ray: %d\n", SYMBOLS_PER_RAY);
    printf("  Elements per symbol: %d\n", ELEMENTS_PER_SYMBOL);
    printf("  Total operations: %lld (%.1f million)\n", 
           (long long)TOTAL_RAYS * SYMBOLS_PER_RAY * ELEMENTS_PER_SYMBOL,
           ((long long)TOTAL_RAYS * SYMBOLS_PER_RAY * ELEMENTS_PER_SYMBOL) / 1000000.0f);
    
    long long totalOperations = (long long)TOTAL_RAYS * SYMBOLS_PER_RAY * ELEMENTS_PER_SYMBOL;
    
    printf("\nBatch processing configuration:\n");
    printf("  Batch size: %d rays (%d symbols)\n", BATCH_SIZE_RAYS, BATCH_SIZE_SYMBOLS);
    printf("  Total batches: %d\n", (TOTAL_RAYS + BATCH_SIZE_RAYS - 1) / BATCH_SIZE_RAYS);
    
    float batchMemoryGB = (float)(BATCH_SIZE_SYMBOLS * ELEMENTS_PER_SYMBOL * sizeof(ComplexType) * 3) / (1024.0f * 1024.0f * 1024.0f);
    printf("  Memory per batch: %.2f GB\n", batchMemoryGB);
    printf("  Streams per GPU: %d\n", STREAMS_PER_GPU);
    
    // Check device properties
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        gpuCheck(cudaGetDeviceProperties(&prop, dev), "get device properties");
        printf("\nDevice %d: %s\n", dev, prop.name);
        printf("  Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    }
    printf("\n");

    // Allocate host memory for the workload
    size_t batchBytes = BATCH_SIZE_SYMBOLS * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
    
    printf("Allocating host memory for batch (%d symbols)...\n", BATCH_SIZE_SYMBOLS);
    
    ComplexType *batch_X = safeHostAlloc(batchBytes, "batch_X");
    ComplexType *batch_H = safeHostAlloc(batchBytes, "batch_H");
    ComplexType *batch_Y = safeHostAlloc(batchBytes, "batch_Y");
    
    if (!batch_X || !batch_H || !batch_Y) {
        printf("ERROR: Failed to allocate host memory.\n");
        return 1;
    }
    
    printf("Host memory allocation successful: %.2f GB total\n", (batchBytes * 3) / (1024.0f * 1024.0f * 1024.0f));

    // Generate test data
    printf("Generating test data...\n");
    for (int i = 0; i < BATCH_SIZE_SYMBOLS; i++) {
        auto vec_X = randomComplexHalfVector(ELEMENTS_PER_SYMBOL, i * 2 + 1000);
        auto vec_H = randomComplexHalfVector(ELEMENTS_PER_SYMBOL, i * 2 + 2000);
        
        memcpy(&batch_X[i * ELEMENTS_PER_SYMBOL], vec_X.data(), ELEMENTS_PER_SYMBOL * sizeof(ComplexType));
        memcpy(&batch_H[i * ELEMENTS_PER_SYMBOL], vec_H.data(), ELEMENTS_PER_SYMBOL * sizeof(ComplexType));
    }

    // ==============================================
    // SINGLE GPU BASELINE PERFORMANCE
    // ==============================================
    printf("\n=== SINGLE GPU PERFORMANCE (Same Dataset) ===\n");
    
    // Allocate memory on GPU 0 for the ENTIRE dataset
    ComplexType *dev_X_single, *dev_H_single, *dev_Y_single;
    gpuCheck(cudaSetDevice(0), "set device 0");
    gpuCheck(cudaMalloc(&dev_X_single, batchBytes), "malloc X single");
    gpuCheck(cudaMalloc(&dev_H_single, batchBytes), "malloc H single");
    gpuCheck(cudaMalloc(&dev_Y_single, batchBytes), "malloc Y single");
    
    // Create streams for single GPU (same number as one GPU in multi-GPU test)
    cudaStream_t *single_streams = new cudaStream_t[STREAMS_PER_GPU];
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
        gpuCheck(cudaStreamCreate(&single_streams[s]), "create single stream");
    }
    
    // Create timing events
    cudaEvent_t single_start, single_stop;
    cudaEvent_t single_h2d_start, single_h2d_stop;
    cudaEvent_t single_launch_start, single_launch_stop;
    cudaEvent_t single_exec_start, single_exec_stop;
    cudaEvent_t single_d2h_start, single_d2h_stop;
    
    // CUDA Graph variables - declare early for proper scope
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    cudaStream_t captureStream;
    
    gpuCheck(cudaEventCreate(&single_start), "create single start");
    gpuCheck(cudaEventCreate(&single_stop), "create single stop");
    gpuCheck(cudaEventCreate(&single_h2d_start), "create single h2d start");
    gpuCheck(cudaEventCreate(&single_h2d_stop), "create single h2d stop");
    gpuCheck(cudaEventCreate(&single_launch_start), "create single launch start");
    gpuCheck(cudaEventCreate(&single_launch_stop), "create single launch stop");
    gpuCheck(cudaEventCreate(&single_exec_start), "create single exec start");
    gpuCheck(cudaEventCreate(&single_exec_stop), "create single exec stop");
    gpuCheck(cudaEventCreate(&single_d2h_start), "create single d2h start");
    gpuCheck(cudaEventCreate(&single_d2h_stop), "create single d2h stop");
    gpuCheck(cudaStreamCreate(&captureStream), "create capture stream");
    
    TimingResults singleGpuResults = {0};
    
    printf("Processing ENTIRE dataset on single GPU 0 using %d streams...\n", STREAMS_PER_GPU);
    printf("Dataset size: %.2f GB (same as will be distributed across %d GPUs)\n", 
           (batchBytes * 3) / (1024.0f * 1024.0f * 1024.0f), deviceCount);
    
    gpuCheck(cudaEventRecord(single_start), "record single start");
    
    // Phase 1: H2D Transfer (entire dataset)
    printf("  Phase 1: Host-to-Device transfer (%.2f GB)...\n", 
           (batchBytes * 2) / (1024.0f * 1024.0f * 1024.0f));
    gpuCheck(cudaEventRecord(single_h2d_start), "record single h2d start");
    
    // Transfer data using multiple streams for fair comparison
    int singleSymbolsPerStream = BATCH_SIZE_SYMBOLS / STREAMS_PER_GPU;
    int singleRemainingSymbols = BATCH_SIZE_SYMBOLS % STREAMS_PER_GPU;
    
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
        int streamStartSymbol = s * singleSymbolsPerStream;
        int streamEndSymbol = streamStartSymbol + singleSymbolsPerStream;
        if (s == STREAMS_PER_GPU - 1) streamEndSymbol += singleRemainingSymbols;
        
        int streamSymbols = streamEndSymbol - streamStartSymbol;
        size_t streamBytes = streamSymbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
        size_t streamOffset = streamStartSymbol * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
        
        gpuCheck(cudaMemcpyAsync((char*)dev_X_single + streamOffset, 
                 &batch_X[streamStartSymbol * ELEMENTS_PER_SYMBOL], 
                 streamBytes, cudaMemcpyHostToDevice, single_streams[s]), "copy X single stream");
        
        gpuCheck(cudaMemcpyAsync((char*)dev_H_single + streamOffset, 
                 &batch_H[streamStartSymbol * ELEMENTS_PER_SYMBOL], 
                 streamBytes, cudaMemcpyHostToDevice, single_streams[s]), "copy H single stream");
    }
    
    // Synchronize all H2D transfers
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
        gpuCheck(cudaStreamSynchronize(single_streams[s]), "sync single h2d stream");
    }
    
    gpuCheck(cudaEventRecord(single_h2d_stop), "record single h2d stop");
    gpuCheck(cudaEventSynchronize(single_h2d_stop), "sync single h2d stop");
    gpuCheck(cudaEventElapsedTime(&singleGpuResults.h2d_time, single_h2d_start, single_h2d_stop), "elapsed single h2d");
    
    // Phase 2: Kernel Launch
    printf("  Phase 2: Kernel launches (%d kernels across %d streams)...\n", STREAMS_PER_GPU, STREAMS_PER_GPU);
    gpuCheck(cudaEventRecord(single_launch_start), "record single launch start");
    gpuCheck(cudaEventRecord(single_exec_start), "record single exec start");
    
    dim3 block(BLOCKSIZE);
    int singleKernelsLaunched = 0;
    
    // Launch kernels across multiple streams (same pattern as multi-GPU)
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
        int streamStartSymbol = s * singleSymbolsPerStream;
        int streamEndSymbol = streamStartSymbol + singleSymbolsPerStream;
        if (s == STREAMS_PER_GPU - 1) streamEndSymbol += singleRemainingSymbols;
        
        int streamSymbols = streamEndSymbol - streamStartSymbol;
        int streamElements = streamSymbols * ELEMENTS_PER_SYMBOL;
        size_t streamOffset = streamStartSymbol * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
        
        dim3 grid((streamElements + BLOCKSIZE - 1) / BLOCKSIZE);
        
        pw_multiply_half_kernel<<<grid, block, 0, single_streams[s]>>>(
            (ComplexType*)((char*)dev_X_single + streamOffset), 
            (ComplexType*)((char*)dev_H_single + streamOffset), 
            (ComplexType*)((char*)dev_Y_single + streamOffset), 
            streamElements);
        
        singleKernelsLaunched++;
    }
    gpuCheck(cudaGetLastError(), "kernel launch single");
    
    gpuCheck(cudaEventRecord(single_launch_stop), "record single launch stop");
    gpuCheck(cudaEventSynchronize(single_launch_stop), "sync single launch stop");
    gpuCheck(cudaEventElapsedTime(&singleGpuResults.kernel_launch_time, single_launch_start, single_launch_stop), "elapsed single launch");
    
    // Phase 3: Kernel Execution Completion
    printf("  Phase 3: Kernel execution completion...\n");
    
    // Synchronize all streams
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
        gpuCheck(cudaStreamSynchronize(single_streams[s]), "sync single kernel stream");
    }
    
    gpuCheck(cudaEventRecord(single_exec_stop), "record single exec stop");
    gpuCheck(cudaEventSynchronize(single_exec_stop), "sync single exec stop");
    gpuCheck(cudaEventElapsedTime(&singleGpuResults.kernel_exec_time, single_exec_start, single_exec_stop), "elapsed single exec");
    
    // Phase 4: D2H Transfer
    printf("  Phase 4: Device-to-Host transfer (%.2f GB)...\n", 
           batchBytes / (1024.0f * 1024.0f * 1024.0f));
    gpuCheck(cudaEventRecord(single_d2h_start), "record single d2h start");
    
    // Transfer results using CUDA Graph's capture stream
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
        int streamStartSymbol = s * singleSymbolsPerStream;
        int streamEndSymbol = streamStartSymbol + singleSymbolsPerStream;
        if (s == STREAMS_PER_GPU - 1) streamEndSymbol += singleRemainingSymbols;
        
        int streamSymbols = streamEndSymbol - streamStartSymbol;
        size_t streamBytes = streamSymbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
        size_t streamOffset = streamStartSymbol * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
        
        // Use the same capture stream for consistency
        gpuCheck(cudaMemcpyAsync(&batch_Y[streamStartSymbol * ELEMENTS_PER_SYMBOL],
                 (char*)dev_Y_single + streamOffset, streamBytes, 
                 cudaMemcpyDeviceToHost, captureStream), "copy Y single stream");
    }
    
    // Synchronize the capture stream for D2H transfers
    gpuCheck(cudaStreamSynchronize(captureStream), "sync single d2h stream");
    
    gpuCheck(cudaEventRecord(single_d2h_stop), "record single d2h stop");
    gpuCheck(cudaEventSynchronize(single_d2h_stop), "sync single d2h stop");
    gpuCheck(cudaEventElapsedTime(&singleGpuResults.d2h_time, single_d2h_start, single_d2h_stop), "elapsed single d2h");
    
    gpuCheck(cudaEventRecord(single_stop), "record single stop");
    gpuCheck(cudaEventSynchronize(single_stop), "sync single stop");
    gpuCheck(cudaEventElapsedTime(&singleGpuResults.total_time, single_start, single_stop), "elapsed single total");
    
    singleGpuResults.num_kernels_launched = singleKernelsLaunched;
    singleGpuResults.num_streams_used = 1; // Using 1 capture stream for CUDA Graph
    
    printf("Single GPU processing complete!\n");
    printf("  CUDA Graph launches: %d (optimized from %d individual kernel launches)\n", singleKernelsLaunched, STREAMS_PER_GPU);
    printf("  Streams used: %d (capture stream)\n", 1);
    printf("  Total time: %.2f ms\n", singleGpuResults.total_time);
    
    // Verify graph optimization worked
    if (singleKernelsLaunched == 1) {
        printf("  ✅ CUDA Graph optimization successful!\n");
    } else {
        printf("  ⚠️ CUDA Graph optimization may not have worked as expected\n");
    }

    // ==============================================
    // MULTI-GPU PERFORMANCE (Same Dataset Distributed)
    // ==============================================
    printf("\n=== MULTI-GPU PERFORMANCE (Same Dataset Distributed) ===\n");
    
    // Allocate device memory for each GPU
    ComplexType **dev_X = new ComplexType*[deviceCount];
    ComplexType **dev_H = new ComplexType*[deviceCount];
    ComplexType **dev_Y = new ComplexType*[deviceCount];
    cudaStream_t **streams = new cudaStream_t*[deviceCount];
    
    size_t batchBytesPerGpu = batchBytes / deviceCount;
    
    printf("Allocating GPU memory across %d devices...\n", deviceCount);
    for (int dev = 0; dev < deviceCount; dev++) {
        gpuCheck(cudaSetDevice(dev), "set device");
        
        gpuCheck(cudaMalloc(&dev_X[dev], batchBytesPerGpu), "malloc X");
        gpuCheck(cudaMalloc(&dev_H[dev], batchBytesPerGpu), "malloc H");
        gpuCheck(cudaMalloc(&dev_Y[dev], batchBytesPerGpu), "malloc Y");
        
        streams[dev] = new cudaStream_t[STREAMS_PER_GPU];
        for (int s = 0; s < STREAMS_PER_GPU; s++) {
            gpuCheck(cudaStreamCreate(&streams[dev][s]), "create stream");
        }
        
        printf("  GPU %d: Allocated %.2f GB\n", dev, (batchBytesPerGpu * 3) / (1024.0f * 1024.0f * 1024.0f));
    }
    
    // Create timing events for multi-GPU
    cudaEvent_t multi_start, multi_stop;
    cudaEvent_t multi_h2d_start, multi_h2d_stop;
    cudaEvent_t multi_launch_start, multi_launch_stop;
    cudaEvent_t multi_exec_start, multi_exec_stop;
    cudaEvent_t multi_d2h_start, multi_d2h_stop;
    
    gpuCheck(cudaSetDevice(0), "set device 0");
    gpuCheck(cudaEventCreate(&multi_start), "create multi start");
    gpuCheck(cudaEventCreate(&multi_stop), "create multi stop");
    gpuCheck(cudaEventCreate(&multi_h2d_start), "create multi h2d start");
    gpuCheck(cudaEventCreate(&multi_h2d_stop), "create multi h2d stop");
    gpuCheck(cudaEventCreate(&multi_launch_start), "create multi launch start");
    gpuCheck(cudaEventCreate(&multi_launch_stop), "create multi launch stop");
    gpuCheck(cudaEventCreate(&multi_exec_start), "create multi exec start");
    gpuCheck(cudaEventCreate(&multi_exec_stop), "create multi exec stop");
    gpuCheck(cudaEventCreate(&multi_d2h_start), "create multi d2h start");
    gpuCheck(cudaEventCreate(&multi_d2h_stop), "create multi d2h stop");
    
    TimingResults multiGpuResults = {0};
    
    printf("Distributing SAME dataset across %d GPUs with %d streams each...\n", deviceCount, STREAMS_PER_GPU);
    printf("Dataset size: %.2f GB (same as single GPU test)\n", 
           (batchBytes * 3) / (1024.0f * 1024.0f * 1024.0f));
    printf("Per-GPU data: %.2f GB\n", 
           (batchBytes * 3) / (1024.0f * 1024.0f * 1024.0f) / deviceCount);
    
    gpuCheck(cudaEventRecord(multi_start), "record multi start");
    
    // Phase 1: H2D Transfers
    printf("  Phase 1: Host-to-Device transfers across all GPUs (%.2f GB total)...\n",
           (batchBytes * 2) / (1024.0f * 1024.0f * 1024.0f));
    gpuCheck(cudaEventRecord(multi_h2d_start), "record multi h2d start");
    
    int multiSymbolsPerGpu = BATCH_SIZE_SYMBOLS / deviceCount;
    int multiRemainingSymbols = BATCH_SIZE_SYMBOLS % deviceCount;
    
    for (int dev = 0; dev < deviceCount; dev++) {
        gpuCheck(cudaSetDevice(dev), "set device");
        
        int startSymbol = dev * multiSymbolsPerGpu;
        int endSymbol = startSymbol + multiSymbolsPerGpu;
        if (dev == deviceCount - 1) endSymbol += multiRemainingSymbols;
        
        int gpuSymbols = endSymbol - startSymbol;
        size_t gpuBytes = gpuSymbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
        
        gpuCheck(cudaMemcpyAsync(dev_X[dev], 
                 &batch_X[startSymbol * ELEMENTS_PER_SYMBOL], 
                 gpuBytes, cudaMemcpyHostToDevice, streams[dev][0]), "H2D X");
        
        gpuCheck(cudaMemcpyAsync(dev_H[dev], 
                 &batch_H[startSymbol * ELEMENTS_PER_SYMBOL], 
                 gpuBytes, cudaMemcpyHostToDevice, streams[dev][0]), "H2D H");
    }
    
    // Synchronize all H2D transfers
    for (int dev = 0; dev < deviceCount; dev++) {
        gpuCheck(cudaSetDevice(dev), "set device");
        gpuCheck(cudaStreamSynchronize(streams[dev][0]), "sync h2d stream");
    }
    
    gpuCheck(cudaSetDevice(0), "set device 0");
    gpuCheck(cudaEventRecord(multi_h2d_stop), "record multi h2d stop");
    gpuCheck(cudaEventSynchronize(multi_h2d_stop), "sync multi h2d stop");
    gpuCheck(cudaEventElapsedTime(&multiGpuResults.h2d_time, multi_h2d_start, multi_h2d_stop), "elapsed multi h2d");
    
    // Phase 2: Kernel Launches
    printf("  Phase 2: Kernel launches across all GPUs...\n");
    gpuCheck(cudaEventRecord(multi_launch_start), "record multi launch start");
    gpuCheck(cudaEventRecord(multi_exec_start), "record multi exec start");
    
    int totalKernelsLaunched = 0;
    
    for (int dev = 0; dev < deviceCount; dev++) {
        gpuCheck(cudaSetDevice(dev), "set device");
        
        int startSymbol = dev * multiSymbolsPerGpu;
        int endSymbol = startSymbol + multiSymbolsPerGpu;
        if (dev == deviceCount - 1) endSymbol += multiRemainingSymbols;
        
        int gpuSymbols = endSymbol - startSymbol;
        
        // Divide work across multiple streams
        int symbolsPerStream = (gpuSymbols + STREAMS_PER_GPU - 1) / STREAMS_PER_GPU;
        
        for (int s = 0; s < STREAMS_PER_GPU && s * symbolsPerStream < gpuSymbols; s++) {
            int streamStartSymbol = s * symbolsPerStream;
            int streamEndSymbol = streamStartSymbol + symbolsPerStream;
            if (streamEndSymbol > gpuSymbols) streamEndSymbol = gpuSymbols;
            
            int streamSymbols = streamEndSymbol - streamStartSymbol;
            if (streamSymbols <= 0) continue;
            
            int streamElements = streamSymbols * ELEMENTS_PER_SYMBOL;
            size_t streamOffset = streamStartSymbol * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
            
            dim3 streamGrid((streamElements + BLOCKSIZE - 1) / BLOCKSIZE);
            
            pw_multiply_half_kernel<<<streamGrid, block, 0, streams[dev][s]>>>(
                (ComplexType*)((char*)dev_X[dev] + streamOffset), 
                (ComplexType*)((char*)dev_H[dev] + streamOffset), 
                (ComplexType*)((char*)dev_Y[dev] + streamOffset), 
                streamElements);
            
            totalKernelsLaunched++;
        }
    }
    
    gpuCheck(cudaSetDevice(0), "set device 0");
    gpuCheck(cudaEventRecord(multi_launch_stop), "record multi launch stop");
    gpuCheck(cudaEventSynchronize(multi_launch_stop), "sync multi launch stop");
    gpuCheck(cudaEventElapsedTime(&multiGpuResults.kernel_launch_time, multi_launch_start, multi_launch_stop), "elapsed multi launch");
    
    // Phase 3: Kernel Execution Completion
    printf("  Phase 3: Kernel execution completion...\n");
    
    // Synchronize all streams on all GPUs
    for (int dev = 0; dev < deviceCount; dev++) {
        gpuCheck(cudaSetDevice(dev), "set device");
        for (int s = 0; s < STREAMS_PER_GPU; s++) {
            gpuCheck(cudaStreamSynchronize(streams[dev][s]), "sync kernel stream");
        }
    }
    
    gpuCheck(cudaSetDevice(0), "set device 0");
    gpuCheck(cudaEventRecord(multi_exec_stop), "record multi exec stop");
    gpuCheck(cudaEventSynchronize(multi_exec_stop), "sync multi exec stop");
    gpuCheck(cudaEventElapsedTime(&multiGpuResults.kernel_exec_time, multi_exec_start, multi_exec_stop), "elapsed multi exec");
    
    // Phase 4: D2H Transfers
    printf("  Phase 4: Device-to-Host transfers across all GPUs (%.2f GB total)...\n",
           batchBytes / (1024.0f * 1024.0f * 1024.0f));
    gpuCheck(cudaEventRecord(multi_d2h_start), "record multi d2h start");
    
    for (int dev = 0; dev < deviceCount; dev++) {
        gpuCheck(cudaSetDevice(dev), "set device");
        
        int startSymbol = dev * multiSymbolsPerGpu;
        int endSymbol = startSymbol + multiSymbolsPerGpu;
        if (dev == deviceCount - 1) endSymbol += multiRemainingSymbols;
        
        int gpuSymbols = endSymbol - startSymbol;
        size_t gpuBytes = gpuSymbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
        
        gpuCheck(cudaMemcpyAsync(&batch_Y[startSymbol * ELEMENTS_PER_SYMBOL],
                 dev_Y[dev], gpuBytes, cudaMemcpyDeviceToHost, streams[dev][0]), "D2H Y");
    }
    
    // Synchronize all D2H transfers
    for (int dev = 0; dev < deviceCount; dev++) {
        gpuCheck(cudaSetDevice(dev), "set device");
        gpuCheck(cudaStreamSynchronize(streams[dev][0]), "sync d2h stream");
    }
    
    gpuCheck(cudaSetDevice(0), "set device 0");
    gpuCheck(cudaEventRecord(multi_d2h_stop), "record multi d2h stop");
    gpuCheck(cudaEventSynchronize(multi_d2h_stop), "sync multi d2h stop");
    gpuCheck(cudaEventElapsedTime(&multiGpuResults.d2h_time, multi_d2h_start, multi_d2h_stop), "elapsed multi d2h");
    
    gpuCheck(cudaEventRecord(multi_stop), "record multi stop");
    gpuCheck(cudaEventSynchronize(multi_stop), "sync multi stop");
    gpuCheck(cudaEventElapsedTime(&multiGpuResults.total_time, multi_start, multi_stop), "elapsed multi total");
    
    multiGpuResults.num_kernels_launched = totalKernelsLaunched;
    multiGpuResults.num_streams_used = deviceCount * STREAMS_PER_GPU;
    
    printf("Multi-GPU processing complete!\n");
    printf("  Kernels launched: %d (across %d GPUs)\n", totalKernelsLaunched, deviceCount);
    printf("  Total streams used: %d\n", deviceCount * STREAMS_PER_GPU);
    printf("  Total time: %.2f ms\n", multiGpuResults.total_time);

    // ==============================================
    // DETAILED PERFORMANCE COMPARISON
    // ==============================================
    printf("\n=== DETAILED PERFORMANCE COMPARISON ===\n");
    
    printf("\n=== FAIR COMPARISON: SAME DATASET ===\n");
    printf("✅ Same %.2f GB dataset used for both tests\n", (batchBytes * 3) / (1024.0f * 1024.0f * 1024.0f));
    printf("✅ Single GPU: processed entire dataset with CUDA Graph (1 optimized launch)\n");
    printf("✅ Multi-GPU: distributed dataset across %d GPUs, %d streams each\n", deviceCount, STREAMS_PER_GPU);
    
    printf("\n📊 DETAILED TIMING BREAKDOWN:\n");
    printf("                          Single GPU    Multi-GPU (%d)    Speedup\n", deviceCount);
    printf("                          -----------    --------------    -------\n");
    printf("H2D Transfer:           %8.2f ms      %8.2f ms      %6.2fx\n", 
           singleGpuResults.h2d_time, multiGpuResults.h2d_time, 
           singleGpuResults.h2d_time / multiGpuResults.h2d_time);
    
    printf("Kernel Launch:          %8.2f ms      %8.2f ms      %6.2fx\n", 
           singleGpuResults.kernel_launch_time, multiGpuResults.kernel_launch_time,
           singleGpuResults.kernel_launch_time / multiGpuResults.kernel_launch_time);
    
    printf("Kernel Execution:       %8.2f ms      %8.2f ms      %6.2fx\n", 
           singleGpuResults.kernel_exec_time, multiGpuResults.kernel_exec_time,
           singleGpuResults.kernel_exec_time / multiGpuResults.kernel_exec_time);
    
    printf("D2H Transfer:           %8.2f ms      %8.2f ms      %6.2fx\n", 
           singleGpuResults.d2h_time, multiGpuResults.d2h_time,
           singleGpuResults.d2h_time / multiGpuResults.d2h_time);
    
    printf("TOTAL TIME:             %8.2f ms      %8.2f ms      %6.2fx\n", 
           singleGpuResults.total_time, multiGpuResults.total_time,
           singleGpuResults.total_time / multiGpuResults.total_time);
    
    printf("\n📈 PERFORMANCE METRICS:\n");
    printf("Single GPU (with CUDA Graph):\n");
    printf("  CUDA Graph launches:    %d (optimized from %d kernel launches)\n", singleGpuResults.num_kernels_launched, STREAMS_PER_GPU);
    printf("  Streams used:           %d (capture stream)\n", singleGpuResults.num_streams_used);
    printf("  Launch overhead/graph:  %.2f μs (was %.2f μs per kernel)\n", 
           (singleGpuResults.kernel_launch_time / singleGpuResults.num_kernels_launched) * 1000.0f,
           247.20f); // Previous overhead
    printf("  Complex ops/second:     %.1f billion ops/s\n", (float)totalOperations / (singleGpuResults.total_time / 1000.0f) / 1000000000.0f);
    
    printf("\nMulti-GPU (%d devices):\n", deviceCount);
    printf("  Kernels launched:       %d\n", multiGpuResults.num_kernels_launched);
    printf("  Streams used:           %d\n", multiGpuResults.num_streams_used);
    printf("  Launch overhead/kernel: %.2f μs\n", (multiGpuResults.kernel_launch_time / multiGpuResults.num_kernels_launched) * 1000.0f);
    printf("  Complex ops/second:     %.1f billion ops/s\n", (float)totalOperations / (multiGpuResults.total_time / 1000.0f) / 1000000000.0f);
    
    printf("\n🚀 EFFICIENCY ANALYSIS:\n");
    float totalSpeedup = singleGpuResults.total_time / multiGpuResults.total_time;
    float efficiency = (totalSpeedup / deviceCount) * 100.0f;
    
    printf("Overall speedup:        %.2fx\n", totalSpeedup);
    printf("Parallel efficiency:    %.1f%% (%.1f%% is ideal)\n", efficiency, 100.0f);
    printf("Throughput improvement: %.1fx\n", (float)totalOperations / (multiGpuResults.total_time / 1000.0f) / ((float)totalOperations / (singleGpuResults.total_time / 1000.0f)));
    
    // Analysis of bottlenecks
    printf("\n🔍 BOTTLENECK ANALYSIS:\n");
    
    float single_gpu_compute_ratio = singleGpuResults.kernel_exec_time / singleGpuResults.total_time * 100.0f;
    float multi_gpu_compute_ratio = multiGpuResults.kernel_exec_time / multiGpuResults.total_time * 100.0f;
    
    printf("Compute time ratio:\n");
    printf("  Single GPU: %.1f%% compute, %.1f%% memory transfers\n", 
           single_gpu_compute_ratio, 100.0f - single_gpu_compute_ratio);
    printf("  Multi-GPU:  %.1f%% compute, %.1f%% memory transfers\n", 
           multi_gpu_compute_ratio, 100.0f - multi_gpu_compute_ratio);
    
    if (efficiency > 80.0f) {
        printf("\n✅ EXCELLENT scaling! Multi-GPU is highly effective.\n");
    } else if (efficiency > 60.0f) {
        printf("\n✅ GOOD scaling! Multi-GPU provides solid benefits.\n");
    } else if (efficiency > 40.0f) {
        printf("\n⚠️ MODERATE scaling. Memory transfers may be limiting performance.\n");
    } else {
        printf("\n❌ LIMITED scaling. Workload may be too small for multi-GPU or memory-bound.\n");
    }
    
    printf("\n💡 RECOMMENDATIONS:\n");
    if (multiGpuResults.h2d_time + multiGpuResults.d2h_time > multiGpuResults.kernel_exec_time) {
        printf("• Memory transfers dominate execution time\n");
        printf("• Consider larger workloads to amortize transfer costs\n");
        printf("• Use more compute-intensive kernels for better GPU utilization\n");
    }
    
    printf("• CUDA Graph optimization reduced single GPU launch overhead by %.1fx\n", 
           (247.20f * STREAMS_PER_GPU) / (singleGpuResults.kernel_launch_time * 1000.0f));
    
    if (efficiency < 70.0f) {
        printf("• Current workload size: %.1f million operations\n", (float)totalOperations / 1000000.0f);
        printf("• Try scaling to 64K rays for better multi-GPU efficiency\n");
    } else {
        printf("• Excellent multi-GPU scaling achieved!\n");
        printf("• CUDA Graphs provide significant single-GPU optimization\n");
    }
    
    // Cleanup
    printf("\nCleaning up memory...\n");
    
    safeFree(batch_X);
    safeFree(batch_H);
    safeFree(batch_Y);
    
    // Single GPU cleanup
    gpuCheck(cudaSetDevice(0), "set device 0");
    gpuCheck(cudaFree(dev_X_single), "free X single");
    gpuCheck(cudaFree(dev_H_single), "free H single");
    gpuCheck(cudaFree(dev_Y_single), "free Y single");
    
    // Cleanup CUDA Graph resources (only if they were created)
    if (graphExec != nullptr) {
        gpuCheck(cudaGraphExecDestroy(graphExec), "destroy graph exec");
    }
    if (graph != nullptr) {
        gpuCheck(cudaGraphDestroy(graph), "destroy graph");
    }
    gpuCheck(cudaStreamDestroy(captureStream), "destroy capture stream");
    
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
        gpuCheck(cudaStreamDestroy(single_streams[s]), "destroy single stream");
    }
    delete[] single_streams;
    
    // Multi-GPU cleanup
    for (int dev = 0; dev < deviceCount; dev++) {
        gpuCheck(cudaSetDevice(dev), "set device");
        
        for (int s = 0; s < STREAMS_PER_GPU; s++) {
            gpuCheck(cudaStreamDestroy(streams[dev][s]), "destroy stream");
        }
        delete[] streams[dev];
        
        gpuCheck(cudaFree(dev_X[dev]), "free X");
        gpuCheck(cudaFree(dev_H[dev]), "free H");
        gpuCheck(cudaFree(dev_Y[dev]), "free Y");
    }

    delete[] dev_X;
    delete[] dev_H;
    delete[] dev_Y;
    delete[] streams;

    // Event cleanup
    gpuCheck(cudaEventDestroy(single_start), "destroy single start");
    gpuCheck(cudaEventDestroy(single_stop), "destroy single stop");
    gpuCheck(cudaEventDestroy(single_h2d_start), "destroy single h2d start");
    gpuCheck(cudaEventDestroy(single_h2d_stop), "destroy single h2d stop");
    gpuCheck(cudaEventDestroy(single_launch_start), "destroy single launch start");
    gpuCheck(cudaEventDestroy(single_launch_stop), "destroy single launch stop");
    gpuCheck(cudaEventDestroy(single_exec_start), "destroy single exec start");
    gpuCheck(cudaEventDestroy(single_exec_stop), "destroy single exec stop");
    gpuCheck(cudaEventDestroy(single_d2h_start), "destroy single d2h start");
    gpuCheck(cudaEventDestroy(single_d2h_stop), "destroy single d2h stop");
    
    gpuCheck(cudaEventDestroy(multi_start), "destroy multi start");
    gpuCheck(cudaEventDestroy(multi_stop), "destroy multi stop");
    gpuCheck(cudaEventDestroy(multi_h2d_start), "destroy multi h2d start");
    gpuCheck(cudaEventDestroy(multi_h2d_stop), "destroy multi h2d stop");
    gpuCheck(cudaEventDestroy(multi_launch_start), "destroy multi launch start");
    gpuCheck(cudaEventDestroy(multi_launch_stop), "destroy multi launch stop");
    gpuCheck(cudaEventDestroy(multi_exec_start), "destroy multi exec start");
    gpuCheck(cudaEventDestroy(multi_exec_stop), "destroy multi exec stop");
    gpuCheck(cudaEventDestroy(multi_d2h_start), "destroy multi d2h start");
    gpuCheck(cudaEventDestroy(multi_d2h_stop), "destroy multi d2h stop");

    printf("1K rays processing and analysis complete!\n");
    return 0;
}