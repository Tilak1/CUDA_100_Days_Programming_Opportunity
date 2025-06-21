#include <stdio.h>
#include <stdlib.h>

#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)

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
  
  printf("=== Multi-GPU Performance Analysis ===\n");
  printf("Found %d CUDA device(s)\n", deviceCount);
  printf("Data size: %ld MB per GPU\n", (FULL_DATA_SIZE * sizeof(int) * 3) / (1024 * 1024));
  printf("Total data size: %ld MB\n", (FULL_DATA_SIZE * sizeof(int) * 3) / (1024 * 1024));
  
  // Check device properties
  for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, dev));
    printf("Device %d: %s\n", dev, prop.name);
    printf("  Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    
    if (!prop.deviceOverlap) {
      printf("  WARNING: Device %d will not handle overlaps!\n", dev);
    }
  }
  printf("\n");

  // Timing events for each GPU
  cudaEvent_t *start_per_gpu = (cudaEvent_t*)malloc(deviceCount * sizeof(cudaEvent_t));
  cudaEvent_t *stop_per_gpu = (cudaEvent_t*)malloc(deviceCount * sizeof(cudaEvent_t));
  cudaEvent_t global_start, global_stop;
  
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

  // Arrays to hold streams and device pointers for each GPU
  cudaStream_t *streams = (cudaStream_t*)malloc(deviceCount * 2 * sizeof(cudaStream_t));
  int **dev_a = (int**)malloc(deviceCount * 2 * sizeof(int*));
  int **dev_b = (int**)malloc(deviceCount * 2 * sizeof(int*));
  int **dev_c = (int**)malloc(deviceCount * 2 * sizeof(int*));

  // Initialize streams and allocate memory on each GPU
  for (int dev = 0; dev < deviceCount; dev++) {
    HANDLE_ERROR(cudaSetDevice(dev));
    
    // Create 2 streams per GPU
    HANDLE_ERROR(cudaStreamCreate(&streams[dev * 2]));
    HANDLE_ERROR(cudaStreamCreate(&streams[dev * 2 + 1]));
    
    // Allocate GPU memory (2 buffers per GPU for double buffering)
    HANDLE_ERROR(cudaMalloc((void**)&dev_a[dev * 2], N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b[dev * 2], N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c[dev * 2], N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_a[dev * 2 + 1], N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b[dev * 2 + 1], N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c[dev * 2 + 1], N * sizeof(int)));
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
  int chunksPerGpu = (FULL_DATA_SIZE / N) / deviceCount;
  int remainingChunks = (FULL_DATA_SIZE / N) % deviceCount;
  
  printf("Work distribution:\n");
  printf("  Total chunks: %d\n", FULL_DATA_SIZE / N);
  printf("  Chunks per GPU: %d\n", chunksPerGpu);
  printf("  Remaining chunks: %d\n", remainingChunks);
  printf("  Streams per GPU: 2\n\n");

  // ==============================================
  // SINGLE GPU BENCHMARK (for comparison)
  // ==============================================
  printf("=== Single GPU Benchmark (GPU 0 only) ===\n");
  HANDLE_ERROR(cudaSetDevice(0));
  
  cudaEvent_t single_start, single_stop;
  HANDLE_ERROR(cudaEventCreate(&single_start));
  HANDLE_ERROR(cudaEventCreate(&single_stop));
  HANDLE_ERROR(cudaEventRecord(single_start, 0));
  
  // Process all data on GPU 0 only
  for (int i = 0; i < FULL_DATA_SIZE; i += N * 2) {
    // Use both streams on GPU 0
    int remaining = FULL_DATA_SIZE - i;
    int chunk1_size = (remaining >= N) ? N : remaining;
    int chunk2_size = (remaining >= N * 2) ? N : ((remaining > N) ? remaining - N : 0);
    
    if (chunk1_size > 0) {
      HANDLE_ERROR(cudaMemcpyAsync(dev_a[0], host_a + i, chunk1_size * sizeof(int), 
                                 cudaMemcpyHostToDevice, streams[0]));
      HANDLE_ERROR(cudaMemcpyAsync(dev_b[0], host_b + i, chunk1_size * sizeof(int), 
                                 cudaMemcpyHostToDevice, streams[0]));
      kernel<<<(chunk1_size + 255) / 256, 256, 0, streams[0]>>>(dev_a[0], dev_b[0], dev_c[0]);
      HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c[0], chunk1_size * sizeof(int), 
                                 cudaMemcpyDeviceToHost, streams[0]));
    }
    
    if (chunk2_size > 0) {
      HANDLE_ERROR(cudaMemcpyAsync(dev_a[1], host_a + i + N, chunk2_size * sizeof(int), 
                                 cudaMemcpyHostToDevice, streams[1]));
      HANDLE_ERROR(cudaMemcpyAsync(dev_b[1], host_b + i + N, chunk2_size * sizeof(int), 
                                 cudaMemcpyHostToDevice, streams[1]));
      kernel<<<(chunk2_size + 255) / 256, 256, 0, streams[1]>>>(dev_a[1], dev_b[1], dev_c[1]);
      HANDLE_ERROR(cudaMemcpyAsync(host_c + i + N, dev_c[1], chunk2_size * sizeof(int), 
                                 cudaMemcpyDeviceToHost, streams[1]));
    }
  }
  
  HANDLE_ERROR(cudaStreamSynchronize(streams[0]));
  HANDLE_ERROR(cudaStreamSynchronize(streams[1]));
  HANDLE_ERROR(cudaEventRecord(single_stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(single_stop));
  
  float singleGpuTime;
  HANDLE_ERROR(cudaEventElapsedTime(&singleGpuTime, single_start, single_stop));
  printf("Single GPU time: %3.1f ms\n\n", singleGpuTime);

  // ==============================================
  // MULTI-GPU BENCHMARK
  // ==============================================
  printf("=== Multi-GPU Benchmark ===\n");
  
  // Start global timing
  HANDLE_ERROR(cudaSetDevice(0));
  HANDLE_ERROR(cudaEventRecord(global_start, 0));
  
  // Start per-GPU timing
  for (int dev = 0; dev < deviceCount; dev++) {
    HANDLE_ERROR(cudaSetDevice(dev));
    HANDLE_ERROR(cudaEventRecord(start_per_gpu[dev], 0));
  }

  // Process data across all GPUs
  for (int dev = 0; dev < deviceCount; dev++) {
    HANDLE_ERROR(cudaSetDevice(dev));
    
    int startChunk = dev * chunksPerGpu;
    int endChunk = (dev + 1) * chunksPerGpu;
    
    // Handle remaining chunks
    if (dev < remainingChunks) {
      endChunk++;
    }
    if (dev == deviceCount - 1) {
      endChunk = FULL_DATA_SIZE / N;
    }
    
    printf("GPU %d processing chunks %d to %d\n", dev, startChunk, endChunk - 1);
    
    for (int chunk = startChunk; chunk < endChunk; chunk += 2) {
      int dataOffset1 = chunk * N;
      int dataOffset2 = (chunk + 1) * N;
      
      // Stream 0 for this GPU
      if (chunk < endChunk) {
        HANDLE_ERROR(cudaMemcpyAsync(dev_a[dev * 2], host_a + dataOffset1, 
                     N * sizeof(int), cudaMemcpyHostToDevice, streams[dev * 2]));
        HANDLE_ERROR(cudaMemcpyAsync(dev_b[dev * 2], host_b + dataOffset1, 
                     N * sizeof(int), cudaMemcpyHostToDevice, streams[dev * 2]));
        kernel<<<N / 256, 256, 0, streams[dev * 2]>>>(dev_a[dev * 2], dev_b[dev * 2], dev_c[dev * 2]);
        HANDLE_ERROR(cudaMemcpyAsync(host_c + dataOffset1, dev_c[dev * 2], 
                     N * sizeof(int), cudaMemcpyDeviceToHost, streams[dev * 2]));
      }
      
      // Stream 1 for this GPU
      if (chunk + 1 < endChunk) {
        HANDLE_ERROR(cudaMemcpyAsync(dev_a[dev * 2 + 1], host_a + dataOffset2, 
                     N * sizeof(int), cudaMemcpyHostToDevice, streams[dev * 2 + 1]));
        HANDLE_ERROR(cudaMemcpyAsync(dev_b[dev * 2 + 1], host_b + dataOffset2, 
                     N * sizeof(int), cudaMemcpyHostToDevice, streams[dev * 2 + 1]));
        kernel<<<N / 256, 256, 0, streams[dev * 2 + 1]>>>(dev_a[dev * 2 + 1], dev_b[dev * 2 + 1], dev_c[dev * 2 + 1]);
        HANDLE_ERROR(cudaMemcpyAsync(host_c + dataOffset2, dev_c[dev * 2 + 1], 
                     N * sizeof(int), cudaMemcpyDeviceToHost, streams[dev * 2 + 1]));
      }
    }
  }

  // Stop per-GPU timing and synchronize
  float *gpuTimes = (float*)malloc(deviceCount * sizeof(float));
  for (int dev = 0; dev < deviceCount; dev++) {
    HANDLE_ERROR(cudaSetDevice(dev));
    HANDLE_ERROR(cudaStreamSynchronize(streams[dev * 2]));
    HANDLE_ERROR(cudaStreamSynchronize(streams[dev * 2 + 1]));
    HANDLE_ERROR(cudaEventRecord(stop_per_gpu[dev], 0));
    HANDLE_ERROR(cudaEventSynchronize(stop_per_gpu[dev]));
    HANDLE_ERROR(cudaEventElapsedTime(&gpuTimes[dev], start_per_gpu[dev], stop_per_gpu[dev]));
  }

  // Stop global timing
  HANDLE_ERROR(cudaSetDevice(0));
  HANDLE_ERROR(cudaEventRecord(global_stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(global_stop));
  
  float totalTime;
  HANDLE_ERROR(cudaEventElapsedTime(&totalTime, global_start, global_stop));

  // ==============================================
  // RESULTS ANALYSIS
  // ==============================================
  printf("\n=== Performance Results ===\n");
  printf("Single GPU time:    %8.1f ms\n", singleGpuTime);
  printf("Multi-GPU total:    %8.1f ms\n", totalTime);
  printf("\nPer-GPU timing:\n");
  
  float maxGpuTime = 0;
  float minGpuTime = gpuTimes[0];
  float avgGpuTime = 0;
  
  for (int dev = 0; dev < deviceCount; dev++) {
    printf("  GPU %d:           %8.1f ms\n", dev, gpuTimes[dev]);
    maxGpuTime = (gpuTimes[dev] > maxGpuTime) ? gpuTimes[dev] : maxGpuTime;
    minGpuTime = (gpuTimes[dev] < minGpuTime) ? gpuTimes[dev] : minGpuTime;
    avgGpuTime += gpuTimes[dev];
  }
  avgGpuTime /= deviceCount;
  
  printf("\nGPU Statistics:\n");
  printf("  Max GPU time:     %8.1f ms\n", maxGpuTime);
  printf("  Min GPU time:     %8.1f ms\n", minGpuTime);
  printf("  Avg GPU time:     %8.1f ms\n", avgGpuTime);
  printf("  GPU time variance: %7.1f ms\n", maxGpuTime - minGpuTime);
  
  printf("\n=== Parallelism Analysis ===\n");
  float speedup = singleGpuTime / totalTime;
  float efficiency = speedup / deviceCount * 100;
  float theoretical_best = singleGpuTime / deviceCount;
  
  printf("Speedup:            %8.2fx\n", speedup);
  printf("Parallel efficiency: %7.1f%%\n", efficiency);
  printf("Theoretical best:   %8.1f ms\n", theoretical_best);
  printf("Overhead:           %8.1f ms\n", totalTime - theoretical_best);
  
  if (totalTime <= maxGpuTime * 1.1) {
    printf("✓ TRUE PARALLELISM ACHIEVED!\n");
    printf("  Total time ≈ slowest GPU time\n");
  } else {
    printf("⚠ LIMITED PARALLELISM\n");
    printf("  Total time > slowest GPU time\n");
  }
  
  if (efficiency > 80) {
    printf("✓ EXCELLENT parallel efficiency (>80%%)\n");
  } else if (efficiency > 60) {
    printf("✓ GOOD parallel efficiency (>60%%)\n");
  } else {
    printf("⚠ POOR parallel efficiency (<60%%)\n");
  }

  // Cleanup
  HANDLE_ERROR(cudaFreeHost(host_a));
  HANDLE_ERROR(cudaFreeHost(host_b));
  HANDLE_ERROR(cudaFreeHost(host_c));

  for (int dev = 0; dev < deviceCount; dev++) {
    HANDLE_ERROR(cudaSetDevice(dev));
    
    HANDLE_ERROR(cudaStreamDestroy(streams[dev * 2]));
    HANDLE_ERROR(cudaStreamDestroy(streams[dev * 2 + 1]));
    
    HANDLE_ERROR(cudaFree(dev_a[dev * 2]));
    HANDLE_ERROR(cudaFree(dev_b[dev * 2]));
    HANDLE_ERROR(cudaFree(dev_c[dev * 2]));
    HANDLE_ERROR(cudaFree(dev_a[dev * 2 + 1]));
    HANDLE_ERROR(cudaFree(dev_b[dev * 2 + 1]));
    HANDLE_ERROR(cudaFree(dev_c[dev * 2 + 1]));
    
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
  HANDLE_ERROR(cudaEventDestroy(single_start));
  HANDLE_ERROR(cudaEventDestroy(single_stop));

  return 0;
}