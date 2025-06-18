#include <cstdio>
 #include <cuda_runtime.h>
 #include <cuComplex.h>
 #include <random>
 #include <vector>
 constexpr int N         = 4096;   // FFT size
 constexpr int BLOCKSIZE = 128;    // 256 threads / block
 /* ------------------------------------------------------------------ */
 /*                            K E R N E L S                           */
 /* ------------------------------------------------------------------ */
 // 1) GLOBAL-memory baseline ---------------------------------------------------
 __global__ void pw_global_kernel(const cuFloatComplex* __restrict__ X,
                                  const cuFloatComplex* __restrict__ H,
                                  cuFloatComplex*       __restrict__ Y,
                                  int                   n)
 {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if (i < n) Y[i] = cuCmulf(X[i], H[i]);
 }
 // 2) SHARED-memory staging ----------------------------------------------------
 __global__ void pw_shared_kernel(const cuFloatComplex* __restrict__ X,
                                  const cuFloatComplex* __restrict__ H,
                                  cuFloatComplex*       __restrict__ Y,
                                  int                   n)
 {
     __shared__ cuFloatComplex Xs[BLOCKSIZE];
     __shared__ cuFloatComplex Hs[BLOCKSIZE];
     int gid = blockIdx.x * blockDim.x + threadIdx.x;
     int tid = threadIdx.x;
     if (gid < n) {
         Xs[tid] = X[gid];
         Hs[tid] = H[gid];
     }
     __syncthreads();
     if (gid < n) Y[gid] = cuCmulf(Xs[tid], Hs[tid]);
 }
 // 3) CONSTANT-memory version --------------------------------------------------
 __constant__ cuFloatComplex H_const[N];
 __global__ void pw_const_kernel(const cuFloatComplex* __restrict__ X,
                                 cuFloatComplex*       __restrict__ Y,
                                 int                   n)
 {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if (i < n) Y[i] = cuCmulf(X[i], H_const[i]);
 }
 /* ------------------------------------------------------------------ */
 /*                    S M A L L   L A U N C H E R S                    */
 /* ------------------------------------------------------------------ */
 inline void launch_pw_global(const cuFloatComplex* X,
                              const cuFloatComplex* H,
                              cuFloatComplex*       Y,
                              int                   n,
                              dim3                  grid,
                              dim3                  block,
                              cudaStream_t          s = 0)
 {
     void* params[] = { (void*)&X, (void*)&H, (void*)&Y, &n };
     cudaLaunchKernel((const void*)pw_global_kernel, grid, block,
                      params, 0, s);
 }
 inline void launch_pw_shared(const cuFloatComplex* X,
                              const cuFloatComplex* H,
                              cuFloatComplex*       Y,
                              int                   n,
                              dim3                  grid,
                              dim3                  block,
                              cudaStream_t          s = 0)
 {
     void* params[] = { (void*)&X, (void*)&H, (void*)&Y, &n };
     cudaLaunchKernel((const void*)pw_shared_kernel, grid, block,
                      params, 0, s);
 }
 inline void launch_pw_const (const cuFloatComplex* X,
                              cuFloatComplex*       Y,
                              int                   n,
                              dim3                  grid,
                              dim3                  block,
                              cudaStream_t          s = 0)
 {
     void* params[] = { (void*)&X, (void*)&Y, &n };
     cudaLaunchKernel((const void*)pw_const_kernel, grid, block,
                      params, 0, s);
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
 std::vector<cuFloatComplex> randomComplexVector(int n)
 {
     std::mt19937 gen(0xC0FFEE);
     std::uniform_real_distribution<float> dist(-1.f, 1.f);
     std::vector<cuFloatComplex> v(n);
     for (auto& c : v) c = make_cuFloatComplex(dist(gen), dist(gen));
     return v;
 }
//  /i
 // ---- Generic timing helper (1 kernel launch) -----------------------
template<typename Launcher, typename... Args>
float timeKernel(dim3 grid, dim3 block, Launcher launch, Args... args)
{
    cudaEvent_t start, stop;
    gpuCheck(cudaEventCreate(&start), "create event");
    gpuCheck(cudaEventCreate(&stop),  "create event");
    gpuCheck(cudaEventRecord(start),  "record start");
    // supply *all* parameters: user args … grid  block  stream(0)
    launch(args..., grid, block, (cudaStream_t)0);
    gpuCheck(cudaEventRecord(stop),   "record stop");
    gpuCheck(cudaEventSynchronize(stop),"sync stop");
    float ms = 0.f;
    gpuCheck(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}
 /* ------------------------------------------------------------------ */
 /*                                M A I N                             */
 /* ------------------------------------------------------------------ */
 int main()
 {
     // 1) Host & device buffers ---------------------------------------
     auto h_X = randomComplexVector(N);
     auto h_H = randomComplexVector(N);
     cuFloatComplex *d_X{}, *d_H{}, *d_Y{};
     size_t bytes = N * sizeof(cuFloatComplex);
     gpuCheck(cudaMalloc(&d_X, bytes), "malloc d_X");
     gpuCheck(cudaMalloc(&d_H, bytes), "malloc d_H");
     gpuCheck(cudaMalloc(&d_Y, bytes), "malloc d_Y");
     gpuCheck(cudaMemcpy(d_X, h_X.data(), bytes, cudaMemcpyHostToDevice), "cpy X");
     gpuCheck(cudaMemcpy(d_H, h_H.data(), bytes, cudaMemcpyHostToDevice), "cpy H");
     // 2) Copy H to constant memory -----------------------------------
     gpuCheck(cudaMemcpyToSymbol(H_const, h_H.data(), bytes), "const copy");
    //  dim3 block(32, 32);   // 32×16 = 512 threads
    //  // Ceil division in each dimension
    // dim3 grid( (N + block.x - 1) / block.x ,
    // (N + block.y - 1) / block.y );
    // // dim3 block(16, 8, 4); // 512 threads in 3-D
    // dim3 grid( (N + block.x - 1) / block.x ,
    // (N + block.y - 1) / block.y );
     dim3 block(BLOCKSIZE);
     dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE);
     // 3) Run & time ---------------------------------------------------
     float t_global = timeKernel(grid, block, launch_pw_global,
                                 d_X, d_H, d_Y, N);
     float t_shared = timeKernel(grid, block, launch_pw_shared,
                                 d_X, d_H, d_Y, N);
     float t_const  = timeKernel(grid, block, launch_pw_const,
                                 d_X,       d_Y, N);
     printf("\n==== 4096-point complex point-wise multiply ====\n");
     printf("Global   memory: %7.3f µs\n", t_global * 1000.0f);
     printf("Shared   memory: %7.3f µs\n", t_shared * 1000.0f);
     printf("Constant memory: %7.3f µs\n", t_const  * 1000.0f);
     cudaFree(d_X); cudaFree(d_H); cudaFree(d_Y);
     return 0;
 }
