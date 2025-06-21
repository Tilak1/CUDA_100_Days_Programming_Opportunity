#include <stdio.h>
#include <stdlib.h>

#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)
#define STREAMS_PER_GPU 512  // Test high stream counts: 32, 64, 128, 256, 512

static void
HandleError (cudaError_t err, const char *file, int line)
{
  if (err != cudaSuccess)
    {
      printf ("%s in %s at line %d\n", cudaGetErrorString (err), file, line);
      exit (EXIT_FAILURE);
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void
kernel (int *a, int *b, int *c)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N)
    {
      int idx1 = (idx + 1) % 256;
      int idx2 = (idx + 2) % 256;
      float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
      float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
      c[idx] = (as + bs) / 2;
    }
}

int
main (void)
{
  int deviceCount;
  HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
  
  if (deviceCount < 1) {
    printf("No CUDA devices found!\n");
    return 1;
  }
  
  printf("=== A100 High-Stream Multi-GPU Analysis ===\n");
  printf("Found %d CUDA device(s)\n", deviceCount);
  printf("Streams per GPU: %d (testing high concurrency)\n", STREAMS_PER_GPU);
  printf("Total streams: %d\n", deviceCount * STREAMS_PER_GPU);
  printf("Data size: %ld MB per GPU\n", (FULL_DATA_SIZE * sizeof(int) * 3) / (1024 * 1024));
  printf("Total data size: %ld MB\n", (FULL_DATA_SIZE * sizeof(int) * 3) / (1024 * 1024));
  
  // Check device properties and enable peer access
  for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, dev));
    printf("Device %d: %s\n", dev, prop.name);
    printf("  Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Concurrent Kernels: %s\n", prop.concurrentKernels ? "YES" : "NO");
    
    if (!prop.deviceOverlap) {
      printf("  WARNING: Device %d will not handle overlaps!\n", dev);
    }
  }
  
  // Enable peer-to-peer access between GPUs (if available)
  printf("\nChecking P2P connectivity:\n");
  for (int dev1 = 0; dev1 < deviceCount; dev1++) {
    for (int dev2 = 0; dev2 < deviceCount; dev2++) {
      if (dev1 != dev2) {
        int canAccessPeer;
        HANDLE_ERROR(cudaDeviceCanAccessPeer(&canAccessPeer, dev1, dev2));
        if (canAccessPeer) {
          HANDLE_ERROR(cudaSetDevice(dev1));
          cudaError_t err = cudaDeviceEnablePeerAccess(dev2, 0);
          if (err == cudaSuccess) {
            printf("  P2P enabled: GPU %d <-> GPU %d\n", dev1, dev2);
          } else if (err != cudaErrorPeerAccessAlreadyEnabled) {
            printf("  P2P failed: GPU %d <-> GPU %d\n", dev1, dev2);
          }
        }
      }
    }
  }
  printf("\n");

  // Timing events
  cudaEvent_t *start_per_gpu = (cudaEvent_t*)malloc(deviceCount * sizeof(cudaEvent_t));
  cudaEvent_t *stop_per_gpu = (cudaEvent_t*)malloc(deviceCount * sizeof(cudaEvent_t));
  cudaEvent_t global_start, global_stop;
  cudaEvent_t launch_start, launch_stop;  // For measuring pure launch time
  cudaEvent_t kernel_launch_start, kernel_launch_stop;  // For pure kernel launch overhead
  cudaEvent_t memcpy_start, memcpy_stop;  // For memory transfer timing
  
  // Create timing events for each GPU
  for (int dev = 0; dev < deviceCount; dev++) {
    HANDLE_ERROR(cudaSetDevice(dev));
    HANDLE_ERROR(cudaEventCreate(&start_per_gpu[dev]));
    HANDLE_ERROR(cudaEventCreate(&stop_per_gpu[dev]));
  }
  
  // Global timing events on device 0
  HANDLE_ERROR(cudaSetDevice(0));
  HANDLE_ERROR(cudaEventCreate(&global_start));
  HANDLE_ERROR(cudaEventCreate(&global_stop));
  HANDLE_ERROR(cudaEventCreate(&launch_start));
  HANDLE_ERROR(cudaEventCreate(&launch_stop));
  HANDLE_ERROR(cudaEventCreate(&kernel_launch_start));
  HANDLE_ERROR(cudaEventCreate(&kernel_launch_stop));
  HANDLE_ERROR(cudaEventCreate(&memcpy_start));
  HANDLE_ERROR(cudaEventCreate(&memcpy_stop));

  // Arrays to hold streams and device pointers for each GPU
  cudaStream_t *streams = (cudaStream_t*)malloc(deviceCount * STREAMS_PER_GPU * sizeof(cudaStream_t));
  int **dev_a = (int**)malloc(deviceCount * STREAMS_PER_GPU * sizeof(int*));
  int **dev_b = (int**)malloc(deviceCount * STREAMS_PER_GPU * sizeof(int*));
  int **dev_c = (int**)malloc(deviceCount * STREAMS_PER_GPU * sizeof(int*));

  // Initialize streams and allocate memory on each GPU
  for (int dev = 0; dev < deviceCount; dev++) {
    HANDLE_ERROR(cudaSetDevice(dev));
    
    // Create STREAMS_PER_GPU streams per GPU
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
      HANDLE_ERROR(cudaStreamCreate(&streams[dev * STREAMS_PER_GPU + s]));
      
      // Allocate GPU memory buffers for each stream
      HANDLE_ERROR(cudaMalloc((void**)&dev_a[dev * STREAMS_PER_GPU + s], N * sizeof(int)));
      HANDLE_ERROR(cudaMalloc((void**)&dev_b[dev * STREAMS_PER_GPU + s], N * sizeof(int)));
      HANDLE_ERROR(cudaMalloc((void**)&dev_c[dev * STREAMS_PER_GPU + s], N * sizeof(int)));
    }
  }

  // Allocate page-locked host memory
  int *host_a, *host_b, *host_c;
  HANDLE_ERROR(cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));

  // Initialize host data
  printf("Initializing data...\n");
  for (int i = 0; i < FULL_DATA_SIZE; i++) {
    host_a[i] = rand();
    host_b[i] = rand();
  }

  // Calculate work distribution
  int totalChunks = FULL_DATA_SIZE / N;
  int chunksPerGpu = totalChunks / deviceCount;
  int remainingChunks = totalChunks % deviceCount;
  
  printf("Work distribution:\n");
  printf("  Total chunks: %d\n", totalChunks);
  printf("  Chunks per GPU: %d\n", chunksPerGpu);
  printf("  Remaining chunks: %d\n", remainingChunks);
  printf("  Streams per GPU: %d\n", STREAMS_PER_GPU);
  printf("  Total streams: %d\n", deviceCount * STREAMS_PER_GPU);
  
  if (chunksPerGpu >= STREAMS_PER_GPU) {
    printf("  Chunks per stream: ~%d\n", chunksPerGpu / STREAMS_PER_GPU);
    printf("  Stream utilization: FULL\n");
  } else {
    printf("  Chunks per stream: <1 (some streams idle)\n");
    printf("  Stream utilization: PARTIAL (%d/%d streams active per GPU)\n", chunksPerGpu, STREAMS_PER_GPU);
  }
  
  printf("  Memory per stream: ~%ld KB\n", (N * sizeof(int) * 3) / 1024);
  printf("\n");

  // ==============================================
  // SEQUENTIAL MULTI-GPU (for comparison) - using only 2 streams
  // ==============================================
  printf("=== Sequential Multi-GPU Launch (2 streams baseline) ===\n");
  
  HANDLE_ERROR(cudaSetDevice(0));
  HANDLE_ERROR(cudaEventRecord(global_start, 0));
  
  // Sequential approach - launch kernels one GPU at a time (using first 2 streams only)
  for (int dev = 0; dev < deviceCount; dev++) {
    HANDLE_ERROR(cudaSetDevice(dev));
    HANDLE_ERROR(cudaEventRecord(start_per_gpu[dev], 0));
    
    int startChunk = dev * chunksPerGpu + (dev < remainingChunks ? dev : remainingChunks);
    int endChunk = startChunk + chunksPerGpu + (dev < remainingChunks ? 1 : 0);
    
    for (int chunk = startChunk; chunk < endChunk; chunk += 2) {
      int dataOffset1 = chunk * N;
      int dataOffset2 = (chunk + 1) * N;
      
      // Stream 0
      if (chunk < endChunk) {
        HANDLE_ERROR(cudaMemcpyAsync(dev_a[dev * STREAMS_PER_GPU], host_a + dataOffset1, 
                     N * sizeof(int), cudaMemcpyHostToDevice, streams[dev * STREAMS_PER_GPU]));
        HANDLE_ERROR(cudaMemcpyAsync(dev_b[dev * STREAMS_PER_GPU], host_b + dataOffset1, 
                     N * sizeof(int), cudaMemcpyHostToDevice, streams[dev * STREAMS_PER_GPU]));
        kernel<<<N / 256, 256, 0, streams[dev * STREAMS_PER_GPU]>>>(dev_a[dev * STREAMS_PER_GPU], dev_b[dev * STREAMS_PER_GPU], dev_c[dev * STREAMS_PER_GPU]);
        HANDLE_ERROR(cudaMemcpyAsync(host_c + dataOffset1, dev_c[dev * STREAMS_PER_GPU], 
                     N * sizeof(int), cudaMemcpyDeviceToHost, streams[dev * STREAMS_PER_GPU]));
      }
      
      // Stream 1
      if (chunk + 1 < endChunk) {
        HANDLE_ERROR(cudaMemcpyAsync(dev_a[dev * STREAMS_PER_GPU + 1], host_a + dataOffset2, 
                     N * sizeof(int), cudaMemcpyHostToDevice, streams[dev * STREAMS_PER_GPU + 1]));
        HANDLE_ERROR(cudaMemcpyAsync(dev_b[dev * STREAMS_PER_GPU + 1], host_b + dataOffset2, 
                     N * sizeof(int), cudaMemcpyHostToDevice, streams[dev * STREAMS_PER_GPU + 1]));
        kernel<<<N / 256, 256, 0, streams[dev * STREAMS_PER_GPU + 1]>>>(dev_a[dev * STREAMS_PER_GPU + 1], dev_b[dev * STREAMS_PER_GPU + 1], dev_c[dev * STREAMS_PER_GPU + 1]);
        HANDLE_ERROR(cudaMemcpyAsync(host_c + dataOffset2, dev_c[dev * STREAMS_PER_GPU + 1], 
                     N * sizeof(int), cudaMemcpyDeviceToHost, streams[dev * STREAMS_PER_GPU + 1]));
      }
    }
    
    HANDLE_ERROR(cudaStreamSynchronize(streams[dev * STREAMS_PER_GPU]));
    HANDLE_ERROR(cudaStreamSynchronize(streams[dev * STREAMS_PER_GPU + 1]));
    HANDLE_ERROR(cudaEventRecord(stop_per_gpu[dev], 0));
  }
  
  HANDLE_ERROR(cudaSetDevice(0));
  HANDLE_ERROR(cudaEventRecord(global_stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(global_stop));
  
  float sequentialTime;
  HANDLE_ERROR(cudaEventElapsedTime(&sequentialTime, global_start, global_stop));
  printf("Sequential multi-GPU time: %8.1f ms\n\n", sequentialTime);

  // Reset result array
  for (int i = 0; i < FULL_DATA_SIZE; i++) {
    host_c[i] = 0;
  }

  // ==============================================
  // SIMULTANEOUS MULTI-GPU LAUNCH WITH MULTIPLE STREAMS
  // ==============================================  
  printf("=== Simultaneous Multi-GPU Launch with %d Streams per GPU ===\n", STREAMS_PER_GPU);
  
  // Pre-copy all data to GPUs first to isolate kernel launch overhead
  printf("Pre-copying data to all GPUs (%d streams per GPU)...\n", STREAMS_PER_GPU);
  HANDLE_ERROR(cudaSetDevice(0));
  HANDLE_ERROR(cudaEventRecord(memcpy_start, 0));
  
  for (int dev = 0; dev < deviceCount; dev++) {
    HANDLE_ERROR(cudaSetDevice(dev));
    
    int startChunk = dev * chunksPerGpu + (dev < remainingChunks ? dev : remainingChunks);
    int endChunk = startChunk + chunksPerGpu + (dev < remainingChunks ? 1 : 0);
    int chunksForThisGpu = endChunk - startChunk;
    
    // Distribute chunks across all streams for this GPU
    for (int chunk = startChunk; chunk < endChunk; chunk++) {
      int streamIdx = (chunk - startChunk) % STREAMS_PER_GPU;
      int bufferIdx = dev * STREAMS_PER_GPU + streamIdx;
      int dataOffset = chunk * N;
      
      // Copy data for this stream
      HANDLE_ERROR(cudaMemcpy(dev_a[bufferIdx], host_a + dataOffset, 
                 N * sizeof(int), cudaMemcpyHostToDevice));
      HANDLE_ERROR(cudaMemcpy(dev_b[bufferIdx], host_b + dataOffset, 
                 N * sizeof(int), cudaMemcpyHostToDevice));
    }
    HANDLE_ERROR(cudaDeviceSynchronize());
  }
  
  HANDLE_ERROR(cudaSetDevice(0));
  HANDLE_ERROR(cudaEventRecord(memcpy_stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(memcpy_stop));
  
  float memcpyTime;
  HANDLE_ERROR(cudaEventElapsedTime(&memcpyTime, memcpy_start, memcpy_stop));
  printf("Data pre-copy time: %8.1f ms\n", memcpyTime);
  
  // Now measure PURE KERNEL LAUNCH overhead AND per-GPU execution
  printf("\nMeasuring pure kernel launch overhead with %d streams per GPU...\n", STREAMS_PER_GPU);
  
  // Start per-GPU timing BEFORE launching kernels
  for (int dev = 0; dev < deviceCount; dev++) {
    HANDLE_ERROR(cudaSetDevice(dev));
    HANDLE_ERROR(cudaEventRecord(start_per_gpu[dev], 0));
  }
  
  HANDLE_ERROR(cudaSetDevice(0));
  HANDLE_ERROR(cudaEventRecord(kernel_launch_start, 0));
  
  // Launch ALL kernels across ALL GPUs and ALL streams
  int totalKernelsLaunched = 0;
  for (int dev = 0; dev < deviceCount; dev++) {
    HANDLE_ERROR(cudaSetDevice(dev));
    
    int startChunk = dev * chunksPerGpu + (dev < remainingChunks ? dev : remainingChunks);
    int endChunk = startChunk + chunksPerGpu + (dev < remainingChunks ? 1 : 0);
    int chunksForThisGpu = endChunk - startChunk;
    
    printf("  GPU %d: launching %d kernels across %d streams\n", dev, chunksForThisGpu, STREAMS_PER_GPU);
    
    // Launch kernels distributed across all streams
    for (int chunk = startChunk; chunk < endChunk; chunk++) {
      int streamIdx = (chunk - startChunk) % STREAMS_PER_GPU;
      int bufferIdx = dev * STREAMS_PER_GPU + streamIdx;
      int streamGlobalIdx = dev * STREAMS_PER_GPU + streamIdx;
      
      kernel<<<N / 256, 256, 0, streams[streamGlobalIdx]>>>(dev_a[bufferIdx], dev_b[bufferIdx], dev_c[bufferIdx]);
      totalKernelsLaunched++;
    }
  }
  
  HANDLE_ERROR(cudaSetDevice(0));
  HANDLE_ERROR(cudaEventRecord(kernel_launch_stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(kernel_launch_stop));
  
  float pureKernelLaunchTime;
  HANDLE_ERROR(cudaEventElapsedTime(&pureKernelLaunchTime, kernel_launch_start, kernel_launch_stop));
  
  printf("Pure kernel launch time: %6.3f ms (%d kernels)\n", pureKernelLaunchTime, totalKernelsLaunched);
  
  // Wait for all kernels to complete and measure execution time
  printf("Waiting for kernel execution to complete...\n");
  HANDLE_ERROR(cudaSetDevice(0));
  HANDLE_ERROR(cudaEventRecord(global_start, 0));
  
  float *gpuTimes = (float*)malloc(deviceCount * sizeof(float));
  for (int dev = 0; dev < deviceCount; dev++) {
    HANDLE_ERROR(cudaSetDevice(dev));
    
    // Synchronize all streams for this GPU
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
      HANDLE_ERROR(cudaStreamSynchronize(streams[dev * STREAMS_PER_GPU + s]));
    }
    
    HANDLE_ERROR(cudaEventRecord(stop_per_gpu[dev], 0));
    HANDLE_ERROR(cudaEventSynchronize(stop_per_gpu[dev]));
    HANDLE_ERROR(cudaEventElapsedTime(&gpuTimes[dev], start_per_gpu[dev], stop_per_gpu[dev]));
  }
  
  HANDLE_ERROR(cudaSetDevice(0));
  HANDLE_ERROR(cudaEventRecord(global_stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(global_stop));
  
  float kernelExecutionTime;
  HANDLE_ERROR(cudaEventElapsedTime(&kernelExecutionTime, global_start, global_stop));
  
  // Copy results back to host using all streams
  printf("Copying results back to host using %d streams per GPU...\n", STREAMS_PER_GPU);
  HANDLE_ERROR(cudaSetDevice(0));
  HANDLE_ERROR(cudaEventRecord(memcpy_start, 0));
  
  for (int dev = 0; dev < deviceCount; dev++) {
    HANDLE_ERROR(cudaSetDevice(dev));
    
    int startChunk = dev * chunksPerGpu + (dev < remainingChunks ? dev : remainingChunks);
    int endChunk = startChunk + chunksPerGpu + (dev < remainingChunks ? 1 : 0);
    
    for (int chunk = startChunk; chunk < endChunk; chunk++) {
      int streamIdx = (chunk - startChunk) % STREAMS_PER_GPU;
      int bufferIdx = dev * STREAMS_PER_GPU + streamIdx;
      int dataOffset = chunk * N;
      
      HANDLE_ERROR(cudaMemcpy(host_c + dataOffset, dev_c[bufferIdx], 
                 N * sizeof(int), cudaMemcpyDeviceToHost));
    }
    HANDLE_ERROR(cudaDeviceSynchronize());
  }
  
  HANDLE_ERROR(cudaSetDevice(0));
  HANDLE_ERROR(cudaEventRecord(memcpy_stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(memcpy_stop));
  
  float resultCopyTime;
  HANDLE_ERROR(cudaEventElapsedTime(&resultCopyTime, memcpy_start, memcpy_stop));
  
  float totalTimeDetailed = memcpyTime + pureKernelLaunchTime + kernelExecutionTime + resultCopyTime;

  // ==============================================
  // DETAILED RESULTS ANALYSIS
  // ==============================================
  printf("\n=== Detailed Performance Breakdown ===\n");
  printf("Sequential multi-GPU:    %8.1f ms\n", sequentialTime);
  printf("\nDetailed simultaneous timing:\n");
  printf("1. Data H2D copy:        %8.1f ms\n", memcpyTime);
  printf("2. Pure kernel launch:   %8.3f ms  ⭐ PURE OVERHEAD\n", pureKernelLaunchTime);
  printf("3. Kernel execution:     %8.1f ms\n", kernelExecutionTime);
  printf("4. Result D2H copy:      %8.1f ms\n", resultCopyTime);
  printf("   Total detailed:       %8.1f ms\n", totalTimeDetailed);
  
  printf("\nPer-GPU execution times:\n");
  float maxGpuTime = 0;
  float minGpuTime = gpuTimes[0];
  float avgGpuTime = 0;
  
  for (int dev = 0; dev < deviceCount; dev++) {
    printf("  GPU %d execution:      %8.1f ms\n", dev, gpuTimes[dev]);
    maxGpuTime = (gpuTimes[dev] > maxGpuTime) ? gpuTimes[dev] : maxGpuTime;
    minGpuTime = (gpuTimes[dev] < minGpuTime) ? gpuTimes[dev] : minGpuTime;
    avgGpuTime += gpuTimes[dev];
  }
  avgGpuTime /= deviceCount;
  
  printf("\n=== Pure Kernel Launch Analysis ===\n");
  int totalKernels = 0;
  for (int dev = 0; dev < deviceCount; dev++) {
    int startChunk = dev * chunksPerGpu + (dev < remainingChunks ? dev : remainingChunks);
    int endChunk = startChunk + chunksPerGpu + (dev < remainingChunks ? 1 : 0);
    totalKernels += ((endChunk - startChunk + 1) / 2) * 2;
  }
  
  float launchOverheadPerKernel = pureKernelLaunchTime / totalKernels;
  float launchOverheadPerGpu = pureKernelLaunchTime / deviceCount;
  
  printf("Total kernels launched:  %8d\n", totalKernels);
  printf("Launch overhead per kernel: %5.1f μs\n", launchOverheadPerKernel * 1000);
  printf("Launch overhead per GPU:    %5.1f μs\n", launchOverheadPerGpu * 1000);
  printf("Kernels per GPU:            %5d\n", totalKernels / deviceCount);
  
  printf("\n=== Parallelism Analysis ===\n");
  float speedup = sequentialTime / totalTimeDetailed;
  float efficiency = speedup / deviceCount * 100;
  
  printf("Overall speedup:         %8.2fx\n", speedup);
  printf("Parallel efficiency:     %8.1f%%\n", efficiency);
  printf("GPU time variance:       %8.1f ms\n", maxGpuTime - minGpuTime);
  printf("Kernel execution sync:   %8.1f ms\n", kernelExecutionTime - maxGpuTime);
  
  // Launch overhead analysis
  if (pureKernelLaunchTime < 0.001) {
    printf("\n🚀 ULTRA-LOW kernel launch overhead (<1μs)!\n");
  } else if (pureKernelLaunchTime < 0.01) {
    printf("\n✓ EXCELLENT kernel launch overhead (<10μs)\n");
  } else if (pureKernelLaunchTime < 0.1) {
    printf("\n✓ GOOD kernel launch overhead (<100μs)\n");
  } else if (pureKernelLaunchTime < 1.0) {
    printf("\n⚠ MODERATE kernel launch overhead (<1ms)\n");
  } else {
    printf("\n❌ HIGH kernel launch overhead (>1ms)\n");
  }
  
  // Per-kernel launch overhead analysis
  if (launchOverheadPerKernel < 0.001) {
    printf("⚡ Per-kernel overhead: <1μs (EXCEPTIONAL)\n");
  } else if (launchOverheadPerKernel < 0.01) {
    printf("✓ Per-kernel overhead: <10μs (EXCELLENT)\n");
  } else if (launchOverheadPerKernel < 0.1) {
    printf("✓ Per-kernel overhead: <100μs (GOOD)\n");
  } else {
    printf("⚠ Per-kernel overhead: >100μs (HIGH)\n");
  }
  
  // Check if kernel execution is truly parallel
  if (kernelExecutionTime <= maxGpuTime * 1.05) {
    printf("✓ PERFECT kernel execution parallelism\n");
  } else {
    printf("⚠ Limited kernel execution parallelism\n");
  }
  
  printf("\n=== A100 Performance Summary ===\n");
  printf("Memory bandwidth utilization: %s\n", 
         (memcpyTime + resultCopyTime) < kernelExecutionTime ? "OPTIMAL" : "BOTTLENECK");
  printf("Compute utilization:          %s\n",
         kernelExecutionTime > (memcpyTime + resultCopyTime) ? "COMPUTE-BOUND" : "MEMORY-BOUND");
  printf("Multi-GPU scaling:            %s\n",
         efficiency > 85 ? "EXCELLENT" : efficiency > 70 ? "GOOD" : "LIMITED");

  // Cleanup
  HANDLE_ERROR(cudaFreeHost(host_a));
  HANDLE_ERROR(cudaFreeHost(host_b));
  HANDLE_ERROR(cudaFreeHost(host_c));

  for (int dev = 0; dev < deviceCount; dev++) {
    HANDLE_ERROR(cudaSetDevice(dev));
    
    for (int s = 0; s < STREAMS_PER_GPU; s++) {
      HANDLE_ERROR(cudaStreamDestroy(streams[dev * STREAMS_PER_GPU + s]));
      HANDLE_ERROR(cudaFree(dev_a[dev * STREAMS_PER_GPU + s]));
      HANDLE_ERROR(cudaFree(dev_b[dev * STREAMS_PER_GPU + s]));
      HANDLE_ERROR(cudaFree(dev_c[dev * STREAMS_PER_GPU + s]));
    }
    
    HANDLE_ERROR(cudaEventDestroy(start_per_gpu[dev]));
    HANDLE_ERROR(cudaEventDestroy(stop_per_gpu[dev]));
  }

  free(streams);
  free(dev_a);
  free(dev_b);
  free(dev_c);
  free(start_per_gpu);
  free(stop_per_gpu);
  free(gpuTimes);

  HANDLE_ERROR(cudaEventDestroy(global_start));
  HANDLE_ERROR(cudaEventDestroy(global_stop));
  HANDLE_ERROR(cudaEventDestroy(launch_start));
  HANDLE_ERROR(cudaEventDestroy(launch_stop));
  HANDLE_ERROR(cudaEventDestroy(kernel_launch_start));
  HANDLE_ERROR(cudaEventDestroy(kernel_launch_stop));
  HANDLE_ERROR(cudaEventDestroy(memcpy_start));
  HANDLE_ERROR(cudaEventDestroy(memcpy_stop));

  return 0;
}