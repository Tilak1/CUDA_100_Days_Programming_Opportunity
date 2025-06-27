// Filename: benchmark_no_graphs_complete.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <random>
#include <vector>
#include <string>
#include <algorithm> // For std::min

constexpr int ELEMENTS_PER_SYMBOL = 4096;
constexpr int SYMBOLS_PER_RAY = 20;
constexpr int TOTAL_RAYS = 1000;
constexpr int BLOCKSIZE = 128;
constexpr int STREAMS_PER_GPU = 4;

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

ComplexType* safeHostAlloc(size_t bytes, const char* name, bool& isPinned) {
    ComplexType* ptr = nullptr;
    if (cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault) == cudaSuccess) {
        printf("  %s: cudaHostAlloc success (%.2f GB) - pinned memory\n", name, bytes / 1e9);
        isPinned = true;
        return ptr;
    }
    printf("  %s: falling back to malloc (%.2f GB) - pageable memory\n", name, bytes / 1e9);
    isPinned = false;
    return (ComplexType*)malloc(bytes);
}

void safeFree(ComplexType* ptr, bool isPinned) {
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
/*                                M A I N                             */
/* ------------------------------------------------------------------ */

int main() {
    int deviceCount;
    gpuCheck(cudaGetDeviceCount(&deviceCount), "get device count");
    if (deviceCount < 1) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    printf("=== Standard Launch Performance Analysis ===\n");
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    long long total_ops_workload = (long long)BATCH_SIZE_SYMBOLS * ELEMENTS_PER_SYMBOL;
    printf("Workload: %.1f million complex multiplies\n\n", total_ops_workload / 1e6);

    // Allocate host memory
    size_t total_bytes = (size_t)BATCH_SIZE_SYMBOLS * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
    bool isPinnedX, isPinnedH, isPinnedY;
    ComplexType *h_X = safeHostAlloc(total_bytes, "h_X", isPinnedX);
    ComplexType *h_H = safeHostAlloc(total_bytes, "h_H", isPinnedH);
    ComplexType *h_Y = safeHostAlloc(total_bytes, "h_Y", isPinnedY);

    // Initialize data
    printf("Initializing data...\n");
    for (int i = 0; i < BATCH_SIZE_SYMBOLS; i++) {
        auto vec_X = randomComplexHalfVector(ELEMENTS_PER_SYMBOL, i * 2);
        auto vec_H = randomComplexHalfVector(ELEMENTS_PER_SYMBOL, i * 2 + 1);
        memcpy(&h_X[(size_t)i * ELEMENTS_PER_SYMBOL], vec_X.data(), 
               ELEMENTS_PER_SYMBOL * sizeof(ComplexType));
        memcpy(&h_H[(size_t)i * ELEMENTS_PER_SYMBOL], vec_H.data(), 
               ELEMENTS_PER_SYMBOL * sizeof(ComplexType));
    }

    // Single-GPU Standard Launch Benchmark
    TimingResults singleGpuResults = {0};
    {
        printf("\n--- Single-GPU Standard Launch Benchmark ---\n");
        gpuCheck(cudaSetDevice(0), "set device 0");
        
        ComplexType *d_X, *d_H, *d_Y;
        gpuCheck(cudaMalloc(&d_X, total_bytes), "malloc d_X");
        gpuCheck(cudaMalloc(&d_H, total_bytes), "malloc d_H");
        gpuCheck(cudaMalloc(&d_Y, total_bytes), "malloc d_Y");

        cudaStream_t* streams = new cudaStream_t[STREAMS_PER_GPU];
        for (int s = 0; s < STREAMS_PER_GPU; s++) gpuCheck(cudaStreamCreate(&streams[s]), "create stream");

        cudaEvent_t start, stop, launch_start, launch_stop;
        gpuCheck(cudaEventCreate(&start), "create event start");
        gpuCheck(cudaEventCreate(&stop), "create event stop");
        gpuCheck(cudaEventCreateWithFlags(&launch_start, cudaEventBlockingSync), "create event launch_start");
        gpuCheck(cudaEventCreateWithFlags(&launch_stop, cudaEventBlockingSync), "create event launch_stop");
        
        // H2D
        gpuCheck(cudaEventRecord(start, 0), "record start H2D");
        for (int s = 0; s < STREAMS_PER_GPU; s++) {
            int start_symbol = s * (BATCH_SIZE_SYMBOLS / STREAMS_PER_GPU);
            int num_symbols = (s == STREAMS_PER_GPU - 1) ? (BATCH_SIZE_SYMBOLS - start_symbol) : (BATCH_SIZE_SYMBOLS / STREAMS_PER_GPU);
            size_t offset = (size_t)start_symbol * ELEMENTS_PER_SYMBOL;
            size_t bytes = (size_t)num_symbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
            gpuCheck(cudaMemcpyAsync(d_X + offset, h_X + offset, bytes, cudaMemcpyHostToDevice, streams[s]), "H2D X");
            gpuCheck(cudaMemcpyAsync(d_H + offset, h_H + offset, bytes, cudaMemcpyHostToDevice, streams[s]), "H2D H");
        }
        for (int s = 0; s < STREAMS_PER_GPU; s++) gpuCheck(cudaStreamSynchronize(streams[s]), "sync H2D stream");
        gpuCheck(cudaEventRecord(stop, 0), "record stop H2D");
        gpuCheck(cudaEventSynchronize(stop), "sync H2D stop event");
        gpuCheck(cudaEventElapsedTime(&singleGpuResults.h2d_time, start, stop), "elapsed H2D");

        // Kernels
        gpuCheck(cudaEventRecord(start, 0), "record kernel exec start");
        gpuCheck(cudaEventRecord(launch_start, 0), "record launch start");
        dim3 block(BLOCKSIZE);
        for (int s = 0; s < STREAMS_PER_GPU; s++) {
            int start_symbol = s * (BATCH_SIZE_SYMBOLS / STREAMS_PER_GPU);
            int num_symbols = (s == STREAMS_PER_GPU - 1) ? (BATCH_SIZE_SYMBOLS - start_symbol) : (BATCH_SIZE_SYMBOLS / STREAMS_PER_GPU);
            int num_elements = num_symbols * ELEMENTS_PER_SYMBOL;
            size_t offset = (size_t)start_symbol * ELEMENTS_PER_SYMBOL;
            dim3 grid((num_elements + BLOCKSIZE - 1) / BLOCKSIZE);
            pw_multiply_half_kernel<<<grid, block, 0, streams[s]>>>(d_X + offset, d_H + offset, d_Y + offset, num_elements);
        }
        gpuCheck(cudaEventRecord(launch_stop, 0), "record launch stop");
        
        for (int s = 0; s < STREAMS_PER_GPU; s++) gpuCheck(cudaStreamSynchronize(streams[s]), "sync kernel stream");
        gpuCheck(cudaEventRecord(stop, 0), "record kernel exec stop");
        
        gpuCheck(cudaEventSynchronize(stop), "sync kernel exec stop event");
        gpuCheck(cudaEventSynchronize(launch_stop), "sync launch stop event");
        
        float total_kernel_time;
        gpuCheck(cudaEventElapsedTime(&total_kernel_time, start, stop), "elapsed total kernel time");
        gpuCheck(cudaEventElapsedTime(&singleGpuResults.kernel_launch_time, launch_start, launch_stop), "kernel launch time");
        singleGpuResults.kernel_exec_time = total_kernel_time - singleGpuResults.kernel_launch_time;
        singleGpuResults.num_kernels_launched = STREAMS_PER_GPU;

        // D2H
        gpuCheck(cudaEventRecord(start, 0), "record d2h start");
        for (int s = 0; s < STREAMS_PER_GPU; s++) {
            int start_symbol = s * (BATCH_SIZE_SYMBOLS / STREAMS_PER_GPU);
            int num_symbols = (s == STREAMS_PER_GPU - 1) ? (BATCH_SIZE_SYMBOLS - start_symbol) : (BATCH_SIZE_SYMBOLS / STREAMS_PER_GPU);
            size_t offset = (size_t)start_symbol * ELEMENTS_PER_SYMBOL;
            size_t bytes = (size_t)num_symbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
            gpuCheck(cudaMemcpyAsync(h_Y + offset, d_Y + offset, bytes, cudaMemcpyDeviceToHost, streams[s]), "D2H");
        }
        for(int s=0; s < STREAMS_PER_GPU; s++) gpuCheck(cudaStreamSynchronize(streams[s]), "sync d2h stream");
        gpuCheck(cudaEventRecord(stop, 0), "record d2h stop");
        gpuCheck(cudaEventSynchronize(stop), "sync d2h stop event");
        gpuCheck(cudaEventElapsedTime(&singleGpuResults.d2h_time, start, stop), "elapsed d2h time");
        
        singleGpuResults.total_time = singleGpuResults.h2d_time + singleGpuResults.kernel_launch_time + singleGpuResults.kernel_exec_time + singleGpuResults.d2h_time;

        gpuCheck(cudaFree(d_X), "free d_X"); gpuCheck(cudaFree(d_H), "free d_H"); gpuCheck(cudaFree(d_Y), "free d_Y");
        for (int s = 0; s < STREAMS_PER_GPU; s++) gpuCheck(cudaStreamDestroy(streams[s]), "destroy stream");
        delete[] streams;
        gpuCheck(cudaEventDestroy(start), "destroy event"); gpuCheck(cudaEventDestroy(stop), "destroy event");
        gpuCheck(cudaEventDestroy(launch_start), "destroy event"); gpuCheck(cudaEventDestroy(launch_stop), "destroy event");
    }

    // Multi-GPU Standard Launch Benchmark
    TimingResults multiGpuResults = {0};
    if (deviceCount > 1) {
        printf("\n--- Multi-GPU Standard Launch Benchmark ---\n");
        
        ComplexType **dev_X = new ComplexType*[deviceCount];
        ComplexType **dev_H = new ComplexType*[deviceCount];
        ComplexType **dev_Y = new ComplexType*[deviceCount];
        cudaStream_t **streams = new cudaStream_t*[deviceCount];
        
        int base_symbols = BATCH_SIZE_SYMBOLS / deviceCount;
        int remainder = BATCH_SIZE_SYMBOLS % deviceCount;
        
        for (int d = 0; d < deviceCount; d++) {
            gpuCheck(cudaSetDevice(d), "set device");
            int symbols_this_gpu = base_symbols + (d < remainder ? 1 : 0);
            size_t bytes = (size_t)symbols_this_gpu * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
            gpuCheck(cudaMalloc(&dev_X[d], bytes), "malloc"); gpuCheck(cudaMalloc(&dev_H[d], bytes), "malloc"); gpuCheck(cudaMalloc(&dev_Y[d], bytes), "malloc");
            streams[d] = new cudaStream_t[STREAMS_PER_GPU];
            for (int s = 0; s < STREAMS_PER_GPU; s++) gpuCheck(cudaStreamCreate(&streams[d][s]), "create stream");
        }
        
        cudaEvent_t start, stop, launch_start, launch_stop;
        gpuCheck(cudaEventCreate(&start), "create event"); gpuCheck(cudaEventCreate(&stop), "create event");
        gpuCheck(cudaEventCreateWithFlags(&launch_start, cudaEventBlockingSync), "create launch start");
        gpuCheck(cudaEventCreateWithFlags(&launch_stop, cudaEventBlockingSync), "create launch stop");
        
        // H2D
        gpuCheck(cudaEventRecord(start, 0), "record h2d start");
        size_t global_offset_symbols = 0;
        for (int d = 0; d < deviceCount; d++) {
            gpuCheck(cudaSetDevice(d), "set device");
            int symbols_this_gpu = base_symbols + (d < remainder ? 1 : 0);
            size_t bytes = (size_t)symbols_this_gpu * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
            gpuCheck(cudaMemcpyAsync(dev_X[d], h_X + global_offset_symbols, bytes, cudaMemcpyHostToDevice, streams[d][0]), "H2D X multi");
            gpuCheck(cudaMemcpyAsync(dev_H[d], h_H + global_offset_symbols, bytes, cudaMemcpyHostToDevice, streams[d][0]), "H2D H multi");
            global_offset_symbols += symbols_this_gpu;
        }
        for (int d = 0; d < deviceCount; d++) { gpuCheck(cudaSetDevice(d), "set dev"); gpuCheck(cudaStreamSynchronize(streams[d][0]), "sync h2d stream"); }
        gpuCheck(cudaEventRecord(stop, 0), "record h2d stop");
        gpuCheck(cudaEventSynchronize(stop), "sync h2d stop event");
        gpuCheck(cudaEventElapsedTime(&multiGpuResults.h2d_time, start, stop), "elapsed h2d");

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
                pw_multiply_half_kernel<<<grid, block, 0, streams[d][s]>>>(dev_X[d] + offset, dev_H[d] + offset, dev_Y[d] + offset, num_elements);
                totalKernels++;
            }
        }
        gpuCheck(cudaEventRecord(launch_stop, 0), "record launch stop");
        for (int d = 0; d < deviceCount; d++) {
            gpuCheck(cudaSetDevice(d), "set device");
            for (int s = 0; s < STREAMS_PER_GPU; s++) gpuCheck(cudaStreamSynchronize(streams[d][s]), "sync kernel stream");
        }
        gpuCheck(cudaEventRecord(stop, 0), "record kernel exec stop");
        gpuCheck(cudaEventSynchronize(stop), "sync kernel exec stop event");
        gpuCheck(cudaEventSynchronize(launch_stop), "sync launch stop event");
        float total_kernel_time;
        gpuCheck(cudaEventElapsedTime(&total_kernel_time, start, stop), "elapsed total kernel time");
        gpuCheck(cudaEventElapsedTime(&multiGpuResults.kernel_launch_time, launch_start, launch_stop), "kernel launch time");
        multiGpuResults.kernel_exec_time = total_kernel_time - multiGpuResults.kernel_launch_time;
        multiGpuResults.num_kernels_launched = totalKernels;

        // D2H
        gpuCheck(cudaEventRecord(start, 0), "record d2h start");
        global_offset_symbols = 0;
        for (int d = 0; d < deviceCount; d++) {
            gpuCheck(cudaSetDevice(d), "set device");
            int symbols_this_gpu = base_symbols + (d < remainder ? 1 : 0);
            size_t bytes = (size_t)symbols_this_gpu * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
            gpuCheck(cudaMemcpyAsync(h_Y + global_offset_symbols, dev_Y[d], bytes, cudaMemcpyDeviceToHost, streams[d][0]), "D2H multi");
            global_offset_symbols += symbols_this_gpu;
        }
        for (int d = 0; d < deviceCount; d++) { gpuCheck(cudaSetDevice(d), "set dev"); gpuCheck(cudaStreamSynchronize(streams[d][0]), "sync d2h stream"); }
        gpuCheck(cudaEventRecord(stop, 0), "record d2h stop");
        gpuCheck(cudaEventSynchronize(stop), "sync d2h stop event");
        gpuCheck(cudaEventElapsedTime(&multiGpuResults.d2h_time, start, stop), "elapsed d2h");

        multiGpuResults.total_time = multiGpuResults.h2d_time + multiGpuResults.kernel_launch_time + multiGpuResults.kernel_exec_time + multiGpuResults.d2h_time;
        
        for (int d = 0; d < deviceCount; d++) {
            gpuCheck(cudaSetDevice(d), "set device");
            gpuCheck(cudaFree(dev_X[d]), "free");
            gpuCheck(cudaFree(dev_H[d]), "free");
            gpuCheck(cudaFree(dev_Y[d]), "free");
            for (int s = 0; s < STREAMS_PER_GPU; s++) gpuCheck(cudaStreamDestroy(streams[d][s]), "destroy stream");
            delete[] streams[d];
        }
        delete[] dev_X; delete[] dev_H; delete[] dev_Y; delete[] streams;
        gpuCheck(cudaEventDestroy(start), "destroy event"); gpuCheck(cudaEventDestroy(stop), "destroy event");
        gpuCheck(cudaEventDestroy(launch_start), "destroy event"); gpuCheck(cudaEventDestroy(launch_stop), "destroy event");
    }
    
    // Final Comparison
    printf("\n=== FINAL PERFORMANCE COMPARISON (Standard Launch) ===\n");
    printf("                          Single GPU");
    if (deviceCount > 1) {
        printf("    Multi-GPU (%d)    Speedup\n", deviceCount);
        printf("                          -----------    --------------    -------\n");
        printf("H2D Time (ms):          %8.2f       %8.2f        %5.2fx\n", 
               singleGpuResults.h2d_time, multiGpuResults.h2d_time, singleGpuResults.h2d_time / multiGpuResults.h2d_time);
        printf("Kernel Launch (ms):     %8.2f       %8.2f        %5.2fx\n", 
               singleGpuResults.kernel_launch_time, multiGpuResults.kernel_launch_time, singleGpuResults.kernel_launch_time / multiGpuResults.kernel_launch_time);
        printf("Kernel Exec (ms):       %8.2f       %8.2f        %5.2fx\n", 
               singleGpuResults.kernel_exec_time, multiGpuResults.kernel_exec_time, singleGpuResults.kernel_exec_time / multiGpuResults.kernel_exec_time);
        printf("D2H Time (ms):          %8.2f       %8.2f        %5.2fx\n", 
               singleGpuResults.d2h_time, multiGpuResults.d2h_time, singleGpuResults.d2h_time / multiGpuResults.d2h_time);
        printf("----------------------------------------------------------\n");
        printf("Total Time (ms):        %8.2f       %8.2f        %5.2fx\n", 
               singleGpuResults.total_time, multiGpuResults.total_time, singleGpuResults.total_time / multiGpuResults.total_time);
        
        float speedup = singleGpuResults.total_time / multiGpuResults.total_time;
        printf("\nOverall Speedup: %.2fx\n", speedup);
        printf("Parallel Efficiency: %.1f%%\n", (speedup / deviceCount) * 100.0);
    } else {
        printf("\n");
        printf("                          -----------\n");
        printf("H2D Time (ms):          %8.2f\n", singleGpuResults.h2d_time);
        printf("Kernel Launch (ms):     %8.2f\n", singleGpuResults.kernel_launch_time);
        printf("Kernel Exec (ms):       %8.2f\n", singleGpuResults.kernel_exec_time);
        printf("D2H Time (ms):          %8.2f\n", singleGpuResults.d2h_time);
        printf("----------------------------------\n");
        printf("Total Time (ms):        %8.2f\n", singleGpuResults.total_time);
        printf("\nNote: Only 1 GPU available, skipping multi-GPU benchmark\n");
    }
    
    safeFree(h_X, isPinnedX);
    safeFree(h_H, isPinnedH);
    safeFree(h_Y, isPinnedY);
    
    return 0;
}