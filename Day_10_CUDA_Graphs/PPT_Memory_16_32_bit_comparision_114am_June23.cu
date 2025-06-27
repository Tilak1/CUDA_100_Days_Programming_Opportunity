// Filename: benchmark_dual_precision.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <random>
#include <vector>
#include <string>
#include <algorithm>

constexpr int ELEMENTS_PER_SYMBOL = 4096;
constexpr int SYMBOLS_PER_RAY = 20;
constexpr int TOTAL_RAYS = 1000;
constexpr int BLOCKSIZE = 128;
constexpr int STREAMS_PER_GPU = 4;

constexpr int BATCH_SIZE_SYMBOLS = TOTAL_RAYS * SYMBOLS_PER_RAY;

/* ------------------------------------------------------------------ */
/*                            K E R N E L S                           */
/* ------------------------------------------------------------------ */

// 16-bit half precision kernel
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

// 32-bit float precision kernel
__global__ void pw_multiply_float_kernel(const cuFloatComplex* __restrict__ X,
                                        const cuFloatComplex* __restrict__ H,
                                        cuFloatComplex* __restrict__ Y,
                                        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Y[i] = cuCmulf(X[i], H[i]);
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

std::vector<cuFloatComplex> randomComplexFloatVector(int n, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<cuFloatComplex> v(n);
    for (auto& c : v) {
        c = make_cuFloatComplex(dist(gen), dist(gen));
    }
    return v;
}

template<typename T>
T* safeHostAlloc(size_t bytes, const char* name, bool& isPinned) {
    T* ptr = nullptr;
    if (cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault) == cudaSuccess) {
        printf("  %s: cudaHostAlloc success (%.2f GB) - pinned memory\n", name, bytes / 1e9);
        isPinned = true;
        return ptr;
    }
    printf("  %s: falling back to malloc (%.2f GB) - pageable memory\n", name, bytes / 1e9);
    isPinned = false;
    return (T*)malloc(bytes);
}

template<typename T>
void safeFree(T* ptr, bool isPinned) {
    if (ptr) {
        if (isPinned) {
            gpuCheck(cudaFreeHost(ptr), "cudaFreeHost");
        } else {
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
/*                   B E N C H M A R K   F U N C T I O N S           */
/* ------------------------------------------------------------------ */

template<typename T>
TimingResults benchmarkSingleGPU(T* h_X, T* h_H, T* h_Y, size_t total_elements, bool is_half) {
    TimingResults results = {0};
    size_t total_bytes = total_elements * sizeof(T);
    
    gpuCheck(cudaSetDevice(0), "set device 0");
    
    T *d_X, *d_H, *d_Y;
    gpuCheck(cudaMalloc(&d_X, total_bytes), "malloc d_X");
    gpuCheck(cudaMalloc(&d_H, total_bytes), "malloc d_H");
    gpuCheck(cudaMalloc(&d_Y, total_bytes), "malloc d_Y");

    cudaStream_t* streams = new cudaStream_t[STREAMS_PER_GPU];
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
        gpuCheck(cudaStreamCreate(&streams[s]), "create stream");
    }

    cudaEvent_t start, stop, launch_start, launch_stop;
    gpuCheck(cudaEventCreate(&start), "create event start");
    gpuCheck(cudaEventCreate(&stop), "create event stop");
    gpuCheck(cudaEventCreateWithFlags(&launch_start, cudaEventBlockingSync), "create event launch_start");
    gpuCheck(cudaEventCreateWithFlags(&launch_stop, cudaEventBlockingSync), "create event launch_stop");
    
    // H2D
    gpuCheck(cudaEventRecord(start, 0), "record start H2D");
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
        int start_symbol = s * (BATCH_SIZE_SYMBOLS / STREAMS_PER_GPU);
        int num_symbols = (s == STREAMS_PER_GPU - 1) ? 
            (BATCH_SIZE_SYMBOLS - start_symbol) : (BATCH_SIZE_SYMBOLS / STREAMS_PER_GPU);
        size_t offset = (size_t)start_symbol * ELEMENTS_PER_SYMBOL;
        size_t bytes = (size_t)num_symbols * ELEMENTS_PER_SYMBOL * sizeof(T);
        gpuCheck(cudaMemcpyAsync(d_X + offset, h_X + offset, bytes, cudaMemcpyHostToDevice, streams[s]), "H2D X");
        gpuCheck(cudaMemcpyAsync(d_H + offset, h_H + offset, bytes, cudaMemcpyHostToDevice, streams[s]), "H2D H");
    }
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
        gpuCheck(cudaStreamSynchronize(streams[s]), "sync H2D stream");
    }
    gpuCheck(cudaEventRecord(stop, 0), "record stop H2D");
    gpuCheck(cudaEventSynchronize(stop), "sync H2D stop event");
    gpuCheck(cudaEventElapsedTime(&results.h2d_time, start, stop), "elapsed H2D");

    // Kernels
    gpuCheck(cudaEventRecord(start, 0), "record kernel exec start");
    gpuCheck(cudaEventRecord(launch_start, 0), "record launch start");
    dim3 block(BLOCKSIZE);
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
        int start_symbol = s * (BATCH_SIZE_SYMBOLS / STREAMS_PER_GPU);
        int num_symbols = (s == STREAMS_PER_GPU - 1) ? 
            (BATCH_SIZE_SYMBOLS - start_symbol) : (BATCH_SIZE_SYMBOLS / STREAMS_PER_GPU);
        int num_elements = num_symbols * ELEMENTS_PER_SYMBOL;
        size_t offset = (size_t)start_symbol * ELEMENTS_PER_SYMBOL;
        dim3 grid((num_elements + BLOCKSIZE - 1) / BLOCKSIZE);
        
        if (is_half) {
            pw_multiply_half_kernel<<<grid, block, 0, streams[s]>>>(
                (__half2*)(d_X + offset), (__half2*)(d_H + offset), (__half2*)(d_Y + offset), num_elements);
        } else {
            pw_multiply_float_kernel<<<grid, block, 0, streams[s]>>>(
                (cuFloatComplex*)(d_X + offset), (cuFloatComplex*)(d_H + offset), 
                (cuFloatComplex*)(d_Y + offset), num_elements);
        }
    }
    gpuCheck(cudaEventRecord(launch_stop, 0), "record launch stop");
    
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
        gpuCheck(cudaStreamSynchronize(streams[s]), "sync kernel stream");
    }
    gpuCheck(cudaEventRecord(stop, 0), "record kernel exec stop");
    
    gpuCheck(cudaEventSynchronize(stop), "sync kernel exec stop event");
    gpuCheck(cudaEventSynchronize(launch_stop), "sync launch stop event");
    
    float total_kernel_time;
    gpuCheck(cudaEventElapsedTime(&total_kernel_time, start, stop), "elapsed total kernel time");
    gpuCheck(cudaEventElapsedTime(&results.kernel_launch_time, launch_start, launch_stop), "kernel launch time");
    results.kernel_exec_time = total_kernel_time - results.kernel_launch_time;
    results.num_kernels_launched = STREAMS_PER_GPU;

    // D2H
    gpuCheck(cudaEventRecord(start, 0), "record d2h start");
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
        int start_symbol = s * (BATCH_SIZE_SYMBOLS / STREAMS_PER_GPU);
        int num_symbols = (s == STREAMS_PER_GPU - 1) ? 
            (BATCH_SIZE_SYMBOLS - start_symbol) : (BATCH_SIZE_SYMBOLS / STREAMS_PER_GPU);
        size_t offset = (size_t)start_symbol * ELEMENTS_PER_SYMBOL;
        size_t bytes = (size_t)num_symbols * ELEMENTS_PER_SYMBOL * sizeof(T);
        gpuCheck(cudaMemcpyAsync(h_Y + offset, d_Y + offset, bytes, cudaMemcpyDeviceToHost, streams[s]), "D2H");
    }
    for(int s = 0; s < STREAMS_PER_GPU; s++) {
        gpuCheck(cudaStreamSynchronize(streams[s]), "sync d2h stream");
    }
    gpuCheck(cudaEventRecord(stop, 0), "record d2h stop");
    gpuCheck(cudaEventSynchronize(stop), "sync d2h stop event");
    gpuCheck(cudaEventElapsedTime(&results.d2h_time, start, stop), "elapsed d2h time");
    
    results.total_time = results.h2d_time + results.kernel_launch_time + 
                        results.kernel_exec_time + results.d2h_time;

    // Cleanup
    gpuCheck(cudaFree(d_X), "free d_X");
    gpuCheck(cudaFree(d_H), "free d_H");
    gpuCheck(cudaFree(d_Y), "free d_Y");
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
        gpuCheck(cudaStreamDestroy(streams[s]), "destroy stream");
    }
    delete[] streams;
    gpuCheck(cudaEventDestroy(start), "destroy event");
    gpuCheck(cudaEventDestroy(stop), "destroy event");
    gpuCheck(cudaEventDestroy(launch_start), "destroy event");
    gpuCheck(cudaEventDestroy(launch_stop), "destroy event");
    
    return results;
}

template<typename T>
TimingResults benchmarkMultiGPU(T* h_X, T* h_H, T* h_Y, size_t total_elements, 
                               int deviceCount, bool is_half) {
    TimingResults results = {0};
    
    T **dev_X = new T*[deviceCount];
    T **dev_H = new T*[deviceCount];
    T **dev_Y = new T*[deviceCount];
    cudaStream_t **streams = new cudaStream_t*[deviceCount];
    
    int base_symbols = BATCH_SIZE_SYMBOLS / deviceCount;
    int remainder = BATCH_SIZE_SYMBOLS % deviceCount;
    
    // Allocate device memory
    for (int d = 0; d < deviceCount; d++) {
        gpuCheck(cudaSetDevice(d), "set device");
        int symbols_this_gpu = base_symbols + (d < remainder ? 1 : 0);
        size_t bytes = (size_t)symbols_this_gpu * ELEMENTS_PER_SYMBOL * sizeof(T);
        gpuCheck(cudaMalloc(&dev_X[d], bytes), "malloc");
        gpuCheck(cudaMalloc(&dev_H[d], bytes), "malloc");
        gpuCheck(cudaMalloc(&dev_Y[d], bytes), "malloc");
        streams[d] = new cudaStream_t[STREAMS_PER_GPU];
        for (int s = 0; s < STREAMS_PER_GPU; s++) {
            gpuCheck(cudaStreamCreate(&streams[d][s]), "create stream");
        }
    }
    
    cudaEvent_t start, stop, launch_start, launch_stop;
    gpuCheck(cudaEventCreate(&start), "create event");
    gpuCheck(cudaEventCreate(&stop), "create event");
    gpuCheck(cudaEventCreateWithFlags(&launch_start, cudaEventBlockingSync), "create launch start");
    gpuCheck(cudaEventCreateWithFlags(&launch_stop, cudaEventBlockingSync), "create launch stop");
    
    // H2D
    gpuCheck(cudaEventRecord(start, 0), "record h2d start");
    size_t global_offset_elements = 0;
    for (int d = 0; d < deviceCount; d++) {
        gpuCheck(cudaSetDevice(d), "set device");
        int symbols_this_gpu = base_symbols + (d < remainder ? 1 : 0);
        size_t elements_this_gpu = (size_t)symbols_this_gpu * ELEMENTS_PER_SYMBOL;
        size_t bytes = elements_this_gpu * sizeof(T);
        gpuCheck(cudaMemcpyAsync(dev_X[d], h_X + global_offset_elements, bytes, 
                               cudaMemcpyHostToDevice, streams[d][0]), "H2D X multi");
        gpuCheck(cudaMemcpyAsync(dev_H[d], h_H + global_offset_elements, bytes, 
                               cudaMemcpyHostToDevice, streams[d][0]), "H2D H multi");
        global_offset_elements += elements_this_gpu;
    }
    for (int d = 0; d < deviceCount; d++) {
        gpuCheck(cudaSetDevice(d), "set dev");
        gpuCheck(cudaStreamSynchronize(streams[d][0]), "sync h2d stream");
    }
    gpuCheck(cudaEventRecord(stop, 0), "record h2d stop");
    gpuCheck(cudaEventSynchronize(stop), "sync h2d stop event");
    gpuCheck(cudaEventElapsedTime(&results.h2d_time, start, stop), "elapsed h2d");

    // Kernels
    gpuCheck(cudaEventRecord(start, 0), "record kernel exec start");
    gpuCheck(cudaEventRecord(launch_start, 0), "record launch start");
    int totalKernels = 0;
    dim3 block(BLOCKSIZE);
    for (int d = 0; d < deviceCount; d++) {
        gpuCheck(cudaSetDevice(d), "set device");
        int symbols_this_gpu = base_symbols + (d < remainder ? 1 : 0);
        int symbols_per_stream = (symbols_this_gpu + STREAMS_PER_GPU - 1) / STREAMS_PER_GPU;
        for (int s = 0; s < STREAMS_PER_GPU; s++) {
            int start_symbol = s * symbols_per_stream;
            int num_symbols = std::min(symbols_per_stream, symbols_this_gpu - start_symbol);
            if (num_symbols <= 0) continue;
            int num_elements = num_symbols * ELEMENTS_PER_SYMBOL;
            size_t offset = (size_t)start_symbol * ELEMENTS_PER_SYMBOL;
            dim3 grid((num_elements + BLOCKSIZE - 1) / BLOCKSIZE);
            
            if (is_half) {
                pw_multiply_half_kernel<<<grid, block, 0, streams[d][s]>>>(
                    (__half2*)(dev_X[d] + offset), (__half2*)(dev_H[d] + offset), 
                    (__half2*)(dev_Y[d] + offset), num_elements);
            } else {
                pw_multiply_float_kernel<<<grid, block, 0, streams[d][s]>>>(
                    (cuFloatComplex*)(dev_X[d] + offset), (cuFloatComplex*)(dev_H[d] + offset), 
                    (cuFloatComplex*)(dev_Y[d] + offset), num_elements);
            }
            totalKernels++;
        }
    }
    gpuCheck(cudaEventRecord(launch_stop, 0), "record launch stop");
    for (int d = 0; d < deviceCount; d++) {
        gpuCheck(cudaSetDevice(d), "set device");
        for (int s = 0; s < STREAMS_PER_GPU; s++) {
            gpuCheck(cudaStreamSynchronize(streams[d][s]), "sync kernel stream");
        }
    }
    gpuCheck(cudaEventRecord(stop, 0), "record kernel exec stop");
    gpuCheck(cudaEventSynchronize(stop), "sync kernel exec stop event");
    gpuCheck(cudaEventSynchronize(launch_stop), "sync launch stop event");
    float total_kernel_time;
    gpuCheck(cudaEventElapsedTime(&total_kernel_time, start, stop), "elapsed total kernel time");
    gpuCheck(cudaEventElapsedTime(&results.kernel_launch_time, launch_start, launch_stop), "kernel launch time");
    results.kernel_exec_time = total_kernel_time - results.kernel_launch_time;
    results.num_kernels_launched = totalKernels;

    // D2H
    gpuCheck(cudaEventRecord(start, 0), "record d2h start");
    global_offset_elements = 0;
    for (int d = 0; d < deviceCount; d++) {
        gpuCheck(cudaSetDevice(d), "set device");
        int symbols_this_gpu = base_symbols + (d < remainder ? 1 : 0);
        size_t elements_this_gpu = (size_t)symbols_this_gpu * ELEMENTS_PER_SYMBOL;
        size_t bytes = elements_this_gpu * sizeof(T);
        gpuCheck(cudaMemcpyAsync(h_Y + global_offset_elements, dev_Y[d], bytes, 
                               cudaMemcpyDeviceToHost, streams[d][0]), "D2H multi");
        global_offset_elements += elements_this_gpu;
    }
    for (int d = 0; d < deviceCount; d++) {
        gpuCheck(cudaSetDevice(d), "set dev");
        gpuCheck(cudaStreamSynchronize(streams[d][0]), "sync d2h stream");
    }
    gpuCheck(cudaEventRecord(stop, 0), "record d2h stop");
    gpuCheck(cudaEventSynchronize(stop), "sync d2h stop event");
    gpuCheck(cudaEventElapsedTime(&results.d2h_time, start, stop), "elapsed d2h");

    results.total_time = results.h2d_time + results.kernel_launch_time + 
                        results.kernel_exec_time + results.d2h_time;
    
    // Cleanup
    for (int d = 0; d < deviceCount; d++) {
        gpuCheck(cudaSetDevice(d), "set device");
        gpuCheck(cudaFree(dev_X[d]), "free");
        gpuCheck(cudaFree(dev_H[d]), "free");
        gpuCheck(cudaFree(dev_Y[d]), "free");
        for (int s = 0; s < STREAMS_PER_GPU; s++) {
            gpuCheck(cudaStreamDestroy(streams[d][s]), "destroy stream");
        }
        delete[] streams[d];
    }
    delete[] dev_X;
    delete[] dev_H;
    delete[] dev_Y;
    delete[] streams;
    gpuCheck(cudaEventDestroy(start), "destroy event");
    gpuCheck(cudaEventDestroy(stop), "destroy event");
    gpuCheck(cudaEventDestroy(launch_start), "destroy event");
    gpuCheck(cudaEventDestroy(launch_stop), "destroy event");
    
    return results;
}

void printComparison(const char* precision, const TimingResults& single, 
                    const TimingResults& multi, int deviceCount) {
    printf("\n=== %s PRECISION PERFORMANCE COMPARISON ===\n", precision);
    printf("                          Single GPU    Multi-GPU (%d)    Speedup\n", deviceCount);
    printf("                          -----------    --------------    -------\n");
    printf("H2D Time (ms):          %8.2f       %8.2f        %5.2fx\n", 
           single.h2d_time, multi.h2d_time, single.h2d_time / multi.h2d_time);
    printf("Kernel Launch (ms):     %8.2f       %8.2f        %5.2fx\n", 
           single.kernel_launch_time, multi.kernel_launch_time, 
           single.kernel_launch_time / multi.kernel_launch_time);
    printf("Kernel Exec (ms):       %8.2f       %8.2f        %5.2fx\n", 
           single.kernel_exec_time, multi.kernel_exec_time, 
           single.kernel_exec_time / multi.kernel_exec_time);
    printf("D2H Time (ms):          %8.2f       %8.2f        %5.2fx\n", 
           single.d2h_time, multi.d2h_time, single.d2h_time / multi.d2h_time);
    printf("----------------------------------------------------------\n");
    printf("Total Time (ms):        %8.2f       %8.2f        %5.2fx\n", 
           single.total_time, multi.total_time, single.total_time / multi.total_time);
    
    float speedup = single.total_time / multi.total_time;
    printf("\nOverall Speedup: %.2fx\n", speedup);
    printf("Parallel Efficiency: %.1f%%\n", (speedup / deviceCount) * 100.0);
}

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
    
    printf("=== Dual Precision CUDA Benchmark ===\n");
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    long long total_ops = (long long)BATCH_SIZE_SYMBOLS * ELEMENTS_PER_SYMBOL;
    printf("Workload: %.1f million complex multiplies\n", total_ops / 1e6);
    printf("Data sizes:\n");
    printf("  16-bit (__half2): %.2f GB\n", (total_ops * sizeof(__half2)) / 1e9);
    printf("  32-bit (cuFloatComplex): %.2f GB\n\n", (total_ops * sizeof(cuFloatComplex)) / 1e9);

    // Allocate host memory for 16-bit
    size_t total_elements = (size_t)BATCH_SIZE_SYMBOLS * ELEMENTS_PER_SYMBOL;
    size_t bytes_half = total_elements * sizeof(__half2);
    size_t bytes_float = total_elements * sizeof(cuFloatComplex);
    
    bool isPinnedX_half, isPinnedH_half, isPinnedY_half;
    bool isPinnedX_float, isPinnedH_float, isPinnedY_float;
    
    printf("Allocating 16-bit host memory:\n");
    __half2 *h_X_half = safeHostAlloc<__half2>(bytes_half, "h_X_half", isPinnedX_half);
    __half2 *h_H_half = safeHostAlloc<__half2>(bytes_half, "h_H_half", isPinnedH_half);
    __half2 *h_Y_half = safeHostAlloc<__half2>(bytes_half, "h_Y_half", isPinnedY_half);
    
    printf("\nAllocating 32-bit host memory:\n");
    cuFloatComplex *h_X_float = safeHostAlloc<cuFloatComplex>(bytes_float, "h_X_float", isPinnedX_float);
    cuFloatComplex *h_H_float = safeHostAlloc<cuFloatComplex>(bytes_float, "h_H_float", isPinnedH_float);
    cuFloatComplex *h_Y_float = safeHostAlloc<cuFloatComplex>(bytes_float, "h_Y_float", isPinnedY_float);

    // Initialize data
    printf("\nInitializing data...\n");
    for (int i = 0; i < BATCH_SIZE_SYMBOLS; i++) {
        // 16-bit data
        auto vec_X_half = randomComplexHalfVector(ELEMENTS_PER_SYMBOL, i * 2);
        auto vec_H_half = randomComplexHalfVector(ELEMENTS_PER_SYMBOL, i * 2 + 1);
        memcpy(&h_X_half[(size_t)i * ELEMENTS_PER_SYMBOL], vec_X_half.data(), 
               ELEMENTS_PER_SYMBOL * sizeof(__half2));
        memcpy(&h_H_half[(size_t)i * ELEMENTS_PER_SYMBOL], vec_H_half.data(), 
               ELEMENTS_PER_SYMBOL * sizeof(__half2));
        
        // 32-bit data
        auto vec_X_float = randomComplexFloatVector(ELEMENTS_PER_SYMBOL, i * 2);
        auto vec_H_float = randomComplexFloatVector(ELEMENTS_PER_SYMBOL, i * 2 + 1);
        memcpy(&h_X_float[(size_t)i * ELEMENTS_PER_SYMBOL], vec_X_float.data(), 
               ELEMENTS_PER_SYMBOL * sizeof(cuFloatComplex));
        memcpy(&h_H_float[(size_t)i * ELEMENTS_PER_SYMBOL], vec_H_float.data(), 
               ELEMENTS_PER_SYMBOL * sizeof(cuFloatComplex));
    }

    // Benchmark 16-bit precision
    printf("\n--- Running 16-bit (__half2) Benchmarks ---\n");
    TimingResults singleGpu16bit = benchmarkSingleGPU(h_X_half, h_H_half, h_Y_half, total_elements, true);
    TimingResults multiGpu16bit = {0};
    if (deviceCount > 1) {
        multiGpu16bit = benchmarkMultiGPU(h_X_half, h_H_half, h_Y_half, total_elements, deviceCount, true);
    }

    // Benchmark 32-bit precision
    printf("\n--- Running 32-bit (cuFloatComplex) Benchmarks ---\n");
    TimingResults singleGpu32bit = benchmarkSingleGPU(h_X_float, h_H_float, h_Y_float, total_elements, false);
    TimingResults multiGpu32bit = {0};
    if (deviceCount > 1) {
        multiGpu32bit = benchmarkMultiGPU(h_X_float, h_H_float, h_Y_float, total_elements, deviceCount, false);
    }

    // Print comparisons
    if (deviceCount > 1) {
        printComparison("16-BIT", singleGpu16bit, multiGpu16bit, deviceCount);
        printComparison("32-BIT", singleGpu32bit, multiGpu32bit, deviceCount);
        
        // Compare 16-bit vs 32-bit performance
        printf("\n=== 16-BIT vs 32-BIT PERFORMANCE COMPARISON ===\n");
        printf("                          16-bit         32-bit         Speedup\n");
        printf("                          -------        -------        -------\n");
        printf("Single GPU (ms):        %8.2f       %8.2f        %5.2fx\n",
               singleGpu16bit.total_time, singleGpu32bit.total_time,
               singleGpu32bit.total_time / singleGpu16bit.total_time);
        printf("Multi-GPU (ms):         %8.2f       %8.2f        %5.2fx\n",
               multiGpu16bit.total_time, multiGpu32bit.total_time,
               multiGpu32bit.total_time / multiGpu16bit.total_time);
        
        float bandwidth_ratio = 2.0f; // 32-bit uses 2x the bandwidth of 16-bit
        printf("\nBandwidth efficiency (16-bit speedup / bandwidth ratio):\n");
        printf("  Single GPU: %.1f%%\n", 
               (singleGpu32bit.total_time / singleGpu16bit.total_time) / bandwidth_ratio * 100);
        printf("  Multi-GPU:  %.1f%%\n", 
               (multiGpu32bit.total_time / multiGpu16bit.total_time) / bandwidth_ratio * 100);
    } else {
        printf("\n=== SINGLE GPU RESULTS ===\n");
        printf("\n16-BIT PRECISION:\n");
        printf("H2D Time (ms):          %8.2f\n", singleGpu16bit.h2d_time);
        printf("Kernel Launch (ms):     %8.2f\n", singleGpu16bit.kernel_launch_time);
        printf("Kernel Exec (ms):       %8.2f\n", singleGpu16bit.kernel_exec_time);
        printf("D2H Time (ms):          %8.2f\n", singleGpu16bit.d2h_time);
        printf("Total Time (ms):        %8.2f\n", singleGpu16bit.total_time);
        
        printf("\n32-BIT PRECISION:\n");
        printf("H2D Time (ms):          %8.2f\n", singleGpu32bit.h2d_time);
        printf("Kernel Launch (ms):     %8.2f\n", singleGpu32bit.kernel_launch_time);
        printf("Kernel Exec (ms):       %8.2f\n", singleGpu32bit.kernel_exec_time);
        printf("D2H Time (ms):          %8.2f\n", singleGpu32bit.d2h_time);
        printf("Total Time (ms):        %8.2f\n", singleGpu32bit.total_time);
        
        printf("\n16-bit vs 32-bit speedup: %.2fx\n", 
               singleGpu32bit.total_time / singleGpu16bit.total_time);
        printf("Note: Only 1 GPU available, skipping multi-GPU benchmark\n");
    }
    
    // Cleanup
    safeFree(h_X_half, isPinnedX_half);
    safeFree(h_H_half, isPinnedH_half);
    safeFree(h_Y_half, isPinnedY_half);
    safeFree(h_X_float, isPinnedX_float);
    safeFree(h_H_float, isPinnedH_float);
    safeFree(h_Y_float, isPinnedY_float);
    
    return 0;
}