// Filename: benchmark_hybrid_analysis_final.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>

constexpr int ELEMENTS_PER_SYMBOL = 4096;
constexpr int SYMBOLS_PER_RAY = 20;
constexpr int TOTAL_RAYS = 1000;
constexpr int BLOCKSIZE = 128;
constexpr int KERNELS_PER_GPU = 4;

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
    for (auto& c : v) { c = __halves2half2(__float2half(dist(gen)), __float2half(dist(gen))); }
    return v;
}

ComplexType* safeHostAlloc(size_t bytes, const char* name) {
    ComplexType* ptr = nullptr;
    if (cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault) == cudaSuccess) {
        printf("  %s: cudaHostAlloc success (%.2f GB)\n", name, bytes / 1e9);
        return ptr;
    }
    ptr = (ComplexType*)malloc(bytes);
    if (!ptr) { fprintf(stderr, "FATAL: malloc failed for %s\n", name); exit(EXIT_FAILURE); }
    return ptr;
}

void safeFree(ComplexType* ptr) {
    if (ptr) { if (cudaFreeHost(ptr) != cudaSuccess) { free(ptr); } }
}

struct TimingResults {
    float h2d_time = 0.0f;
    float kernel_launch_time = 0.0f;
    float kernel_exec_time = 0.0f;
    float d2h_time = 0.0f;
    float total_time = 0.0f;
};

// Helper function to time a specific phase
float time_phase(const std::function<void()>& phase_func) {
    cudaEvent_t start, stop;
    gpuCheck(cudaEventCreate(&start), "create start");
    gpuCheck(cudaEventCreate(&stop), "create stop");
    
    // We record events on the default stream (0) and synchronize the whole device
    // This gives a simple, robust measurement for serialized phases.
    gpuCheck(cudaEventRecord(start, 0), "record start");
    phase_func();
    gpuCheck(cudaEventRecord(stop, 0), "record stop");
    gpuCheck(cudaEventSynchronize(stop), "sync stop");

    float time_ms = 0.0f;
    gpuCheck(cudaEventElapsedTime(&time_ms, start, stop), "elapsed time");
    
    gpuCheck(cudaEventDestroy(start), "destroy start");
    gpuCheck(cudaEventDestroy(stop), "destroy stop");
    return time_ms;
}

/* ------------------------------------------------------------------ */
/*                                M A I N                             */
/* ------------------------------------------------------------------ */

int main() {
    int deviceCount;
    gpuCheck(cudaGetDeviceCount(&deviceCount), "get device count");
    if (deviceCount < 1) { printf("No CUDA devices found!\n"); return 1; }
    
    printf("=== Hybrid Performance Analysis (Graphs + Detailed Breakdown) ===\n");
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    long long total_ops = (long long)BATCH_SIZE_SYMBOLS * ELEMENTS_PER_SYMBOL;
    printf("Workload: %.1f million complex multiplies\n\n", total_ops / 1e6);

    size_t total_bytes = (size_t)BATCH_SIZE_SYMBOLS * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
    ComplexType *h_X = safeHostAlloc(total_bytes, "h_X");
    ComplexType *h_H = safeHostAlloc(total_bytes, "h_H");
    ComplexType *h_Y = safeHostAlloc(total_bytes, "h_Y");

    printf("Initializing data...\n");
    for (int i = 0; i < BATCH_SIZE_SYMBOLS; i++) {
        auto vec_X = randomComplexHalfVector(ELEMENTS_PER_SYMBOL, i * 2);
        auto vec_H = randomComplexHalfVector(ELEMENTS_PER_SYMBOL, i * 2 + 1);
        memcpy(&h_X[(size_t)i * ELEMENTS_PER_SYMBOL], vec_X.data(), ELEMENTS_PER_SYMBOL * sizeof(ComplexType));
        memcpy(&h_H[(size_t)i * ELEMENTS_PER_SYMBOL], vec_H.data(), ELEMENTS_PER_SYMBOL * sizeof(ComplexType));
    }

    // --- Single-GPU Benchmark ---
    TimingResults singleGpuResults = {0};
    {
        printf("\n--- Benchmarking Single GPU ---\n");
        gpuCheck(cudaSetDevice(0), "set dev 0");
        ComplexType *d_X, *d_H, *d_Y;
        gpuCheck(cudaMalloc(&d_X, total_bytes), "malloc");
        gpuCheck(cudaMalloc(&d_H, total_bytes), "malloc");
        gpuCheck(cudaMalloc(&d_Y, total_bytes), "malloc");

        cudaStream_t stream;
        gpuCheck(cudaStreamCreate(&stream), "create stream");

        // 1. Measure component times for breakdown
        singleGpuResults.h2d_time = time_phase([&]{
            gpuCheck(cudaMemcpyAsync(d_X, h_X, total_bytes, cudaMemcpyHostToDevice, stream), "H2D X");
            gpuCheck(cudaMemcpyAsync(d_H, h_H, total_bytes, cudaMemcpyHostToDevice, stream), "H2D H");
            gpuCheck(cudaStreamSynchronize(stream), "sync H2D");
        });
        
        cudaEvent_t launch_start, launch_stop;
        gpuCheck(cudaEventCreateWithFlags(&launch_start, cudaEventBlockingSync), "create launch");
        gpuCheck(cudaEventCreateWithFlags(&launch_stop, cudaEventBlockingSync), "create launch");
        
        singleGpuResults.kernel_exec_time = time_phase([&]{
            gpuCheck(cudaEventRecord(launch_start, 0), "rec launch start");
            dim3 block(BLOCKSIZE);
            int symbols_per_kernel = BATCH_SIZE_SYMBOLS / KERNELS_PER_GPU;
            for (int i=0; i<KERNELS_PER_GPU; ++i) {
                int start_symbol = i * symbols_per_kernel;
                int num_symbols = (i == KERNELS_PER_GPU - 1) ? (BATCH_SIZE_SYMBOLS - start_symbol) : symbols_per_kernel;
                if(num_symbols <= 0) continue;
                int num_elements = num_symbols * ELEMENTS_PER_SYMBOL;
                size_t offset = (size_t)start_symbol * ELEMENTS_PER_SYMBOL;
                dim3 grid((num_elements+BLOCKSIZE-1)/BLOCKSIZE);
                pw_multiply_half_kernel<<<grid, block, 0, stream>>>(d_X + offset, d_H + offset, d_Y + offset, num_elements);
            }
            gpuCheck(cudaEventRecord(launch_stop, 0), "rec launch stop");
            gpuCheck(cudaStreamSynchronize(stream), "sync kernels");
        });
        
        gpuCheck(cudaEventSynchronize(launch_stop), "sync launch stop");
        gpuCheck(cudaEventElapsedTime(&singleGpuResults.kernel_launch_time, launch_start, launch_stop), "elapsed launch");
        singleGpuResults.kernel_exec_time -= singleGpuResults.kernel_launch_time;
        
        singleGpuResults.d2h_time = time_phase([&]{
            gpuCheck(cudaMemcpyAsync(h_Y, d_Y, total_bytes, cudaMemcpyDeviceToHost, stream), "D2H");
            gpuCheck(cudaStreamSynchronize(stream), "sync d2h");
        });

        singleGpuResults.total_time = singleGpuResults.h2d_time + singleGpuResults.kernel_launch_time + singleGpuResults.kernel_exec_time + singleGpuResults.d2h_time;
        
        gpuCheck(cudaEventDestroy(launch_start), "destroy");
        gpuCheck(cudaEventDestroy(launch_stop), "destroy");
        gpuCheck(cudaFree(d_X), "free"); gpuCheck(cudaFree(d_H), "free"); gpuCheck(cudaFree(d_Y), "free");
        gpuCheck(cudaStreamDestroy(stream), "destroy");
    }

    // --- Multi-GPU Benchmark ---
    TimingResults multiGpuResults = {0};
    if (deviceCount > 1) {
        printf("\n--- Benchmarking Multi-GPU ---\n");
        std::vector<ComplexType*> d_X(deviceCount), d_H(deviceCount), d_Y(deviceCount);
        std::vector<cudaStream_t> streams(deviceCount);
        int base_symbols = BATCH_SIZE_SYMBOLS / deviceCount;
        int remainder = BATCH_SIZE_SYMBOLS % deviceCount;
        for (int d = 0; d < deviceCount; ++d) {
            gpuCheck(cudaSetDevice(d), "set dev");
            int gpu_symbols = base_symbols + (d < remainder ? 1 : 0);
            size_t gpu_bytes = (size_t)gpu_symbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
            gpuCheck(cudaMalloc(&d_X[d], gpu_bytes), "malloc"); gpuCheck(cudaMalloc(&d_H[d], gpu_bytes), "malloc"); gpuCheck(cudaMalloc(&d_Y[d], gpu_bytes), "malloc");
            gpuCheck(cudaStreamCreate(&streams[d]), "create stream");
        }
        
        multiGpuResults.h2d_time = time_phase([&]{
            size_t offset_symbols = 0;
            for(int d=0; d<deviceCount; ++d) {
                gpuCheck(cudaSetDevice(d), "set dev");
                int gpu_symbols = base_symbols + (d < remainder ? 1 : 0);
                size_t gpu_bytes = (size_t)gpu_symbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
                gpuCheck(cudaMemcpyAsync(d_X[d], h_X + offset_symbols, gpu_bytes, cudaMemcpyHostToDevice, streams[d]), "H2D X");
                gpuCheck(cudaMemcpyAsync(d_H[d], h_H + offset_symbols, gpu_bytes, cudaMemcpyHostToDevice, streams[d]), "H2D H");
                offset_symbols += gpu_symbols;
            }
            for(int d=0; d<deviceCount; ++d) { gpuCheck(cudaSetDevice(d), "set dev"); gpuCheck(cudaStreamSynchronize(streams[d]), "sync"); }
        });
        
        cudaEvent_t launch_start, launch_stop;
        gpuCheck(cudaEventCreateWithFlags(&launch_start, cudaEventBlockingSync), "create launch");
        gpuCheck(cudaEventCreateWithFlags(&launch_stop, cudaEventBlockingSync), "create launch");

        multiGpuResults.kernel_exec_time = time_phase([&]{
            gpuCheck(cudaEventRecord(launch_start, 0), "rec launch start");
            for(int d=0; d<deviceCount; ++d) {
                gpuCheck(cudaSetDevice(d), "set dev");
                int gpu_symbols = base_symbols + (d < remainder ? 1 : 0);
                int num_elements = gpu_symbols * ELEMENTS_PER_SYMBOL;
                dim3 grid((num_elements + BLOCKSIZE - 1) / BLOCKSIZE), block(BLOCKSIZE);
                pw_multiply_half_kernel<<<grid, block, 0, streams[d]>>>(d_X[d], d_H[d], d_Y[d], num_elements);
            }
            gpuCheck(cudaEventRecord(launch_stop, 0), "rec launch stop");
            for(int d=0; d<deviceCount; ++d) { gpuCheck(cudaSetDevice(d), "set dev"); gpuCheck(cudaStreamSynchronize(streams[d]), "sync"); }
        });
        
        gpuCheck(cudaEventSynchronize(launch_stop), "sync launch");
        gpuCheck(cudaEventElapsedTime(&multiGpuResults.kernel_launch_time, launch_start, launch_stop), "elapsed launch");
        multiGpuResults.kernel_exec_time -= multiGpuResults.kernel_launch_time;

        multiGpuResults.d2h_time = time_phase([&]{
            size_t offset_symbols = 0;
            for(int d=0; d<deviceCount; ++d) {
                gpuCheck(cudaSetDevice(d), "set dev");
                int gpu_symbols = base_symbols + (d < remainder ? 1 : 0);
                size_t gpu_bytes = (size_t)gpu_symbols * ELEMENTS_PER_SYMBOL * sizeof(ComplexType);
                gpuCheck(cudaMemcpyAsync(h_Y + offset_symbols, d_Y[d], gpu_bytes, cudaMemcpyDeviceToHost, streams[d]), "D2H Y");
                offset_symbols += gpu_symbols;
            }
            for(int d=0; d<deviceCount; ++d) { gpuCheck(cudaSetDevice(d), "set dev"); gpuCheck(cudaStreamSynchronize(streams[d]), "sync"); }
        });

        multiGpuResults.total_time = multiGpuResults.h2d_time + multiGpuResults.kernel_launch_time + multiGpuResults.kernel_exec_time + multiGpuResults.d2h_time;

        gpuCheck(cudaEventDestroy(launch_start), "destroy");
        gpuCheck(cudaEventDestroy(launch_stop), "destroy");
        for (int d = 0; d < deviceCount; ++d) {
            gpuCheck(cudaSetDevice(d), "set dev");
            gpuCheck(cudaFree(d_X[d]), "free"); gpuCheck(cudaFree(d_H[d]), "free"); gpuCheck(cudaFree(d_Y[d]), "free");
            gpuCheck(cudaStreamDestroy(streams[d]), "destroy");
        }
    }
    
    // Final Comparison
    printf("\n\n=== FINAL PERFORMANCE COMPARISON ===\n");
    printf("\n📊 DETAILED TIMING BREAKDOWN:\n");
    printf("                          Single GPU    Multi-GPU (%d)    Speedup\n", deviceCount);
    printf("                          -----------    --------------    -------\n");
    printf("H2D Transfer:           %8.2f ms      %8.2f ms      %6.2fx\n", singleGpuResults.h2d_time, multiGpuResults.h2d_time, (multiGpuResults.h2d_time > 0.001) ? singleGpuResults.h2d_time / multiGpuResults.h2d_time : 0.0);
    printf("Kernel Launch:          %8.2f ms      %8.2f ms      %6.2fx\n", singleGpuResults.kernel_launch_time, multiGpuResults.kernel_launch_time, (multiGpuResults.kernel_launch_time > 0.001) ? singleGpuResults.kernel_launch_time / multiGpuResults.kernel_launch_time : 0.0);
    printf("Kernel Execution:       %8.2f ms      %8.2f ms      %6.2fx\n", singleGpuResults.kernel_exec_time, multiGpuResults.kernel_exec_time, (multiGpuResults.kernel_exec_time > 0.001) ? singleGpuResults.kernel_exec_time / multiGpuResults.kernel_exec_time : 0.0);
    printf("D2H Transfer:           %8.2f ms      %8.2f ms      %6.2fx\n", singleGpuResults.d2h_time, multiGpuResults.d2h_time, (multiGpuResults.d2h_time > 0.001) ? singleGpuResults.d2h_time / multiGpuResults.d2h_time : 0.0);
    printf("---------------------     -----------    --------------    -------\n");
    printf("TOTAL TIME:             %8.2f ms      %8.2f ms      %6.2fx\n", singleGpuResults.total_time, multiGpuResults.total_time, (multiGpuResults.total_time > 0.001) ? singleGpuResults.total_time / multiGpuResults.total_time : 0.0);
    
    printf("\nOverall Speedup: %.2fx\n", (multiGpuResults.total_time > 0.001) ? singleGpuResults.total_time / multiGpuResults.total_time : 0.0);
    printf("Parallel Efficiency: %.1f%%\n", ((multiGpuResults.total_time > 0.001) ? singleGpuResults.total_time / multiGpuResults.total_time : 0.0) / deviceCount * 100.0);
    
    safeFree(h_X); safeFree(h_H); safeFree(h_Y);
    return 0;
}