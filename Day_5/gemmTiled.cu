#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 256
#define TILE_WIDTH 2  // Tile width for shared memory


__global__ void gemmGPUTiledkernel(int *a, int *b, int *c, int n)
{    
    __shared__ int T_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ int T_b[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;



    
    int sum = 0;
    
    // No of tiles = n/TILE_WIDTH
     //incremented way of loading the n/TILE_WIDTH tiles / that many phases 
    // Here, we are iterating over the tiles of the input matrices.

    for (int ph=0; ph<n/TILE_WIDTH; ++ph)
    {   
        //Typically to access matrix elements in a 2D array, we use the formula:
        //T_b[ty][tx] = b[row*width][k] - // [row*width] for horizontal memory placing in the memory 
        //But here in a tiled way - k = ph*TILE_WIDTH + tx;  // Load B tile into shared memory
        // Here ph*TILE_WIDTH is the row index of the tile, and tx is the column index within the tile.
        // Converting 2d thinking to 1d thinking

        int global_col = ph * TILE_WIDTH + tx;  // Calculate the global column index for the tile
        int global_row = ph * TILE_WIDTH + ty;  // Calculate the global row index for the tile

        // Since we use row major access - ty here refers rows and tx refers columns in the tile.
        // Hence it is T_a [ty][tx]
        T_a[ty][tx] = (row < n && global_col < n) ? a[row * n + (global_col)] : 0;  // Load A tile into shared memory
        T_b[ty][tx] = (global_row < n && col < n) ? b[(global_row) * n + col] : 0;

        //T_b[ty][tx] = b[k*width][col] - [k*width] for horizontal memory placing - as we need to skip width number of elements to get to the next row in the matrix.

        //k = ph*TILE_WIDTH + ty;  // Load B tile into shared memory
        //..Here again ph*TILE_WIDTH is the row index of the tile, and ty (since accumulating elements vertically) is the row index within the tile.
        
        //T_b[ty][tx] = b[(ph * TILE_WIDTH + ty) * n + col];  // Load B tile into shared memory
        
        __syncthreads();  // Ensure all threads have loaded their tiles before proceeding

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += T_a[ty][k] * T_b[k][tx];  // Perform the multiplication and accumulate the sum
        }
        __syncthreads();  // Ensure all threads have completed their computations before proceeding to the next tile

    }

    // After processing all tiles, write the result to the output matrix
    // Here, we check if the row and column indices are within bounds of the output matrix
    // and then write the accumulated sum to the output matrix.
    // This is important to avoid writing out of bounds, especially when the matrix size is not a perfect multiple of TILE_WIDTH.
    // Note: The row and col indices are calculated based on the block and thread indices.
    // This ensures that each thread writes to the correct position in the output matrix.

    if ((row < n) && (col < n)) {
            c[row * n + col] = sum;  // Write the accumulated sum to the output matrix
        }
}


__global__ void gemmGPUkernel(int *a, int *b, int *c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < n) && (col < n)) {
        int sum = 0;
        for (int k = 0; k < n; ++k)
            sum += a[row * n + k] * b[k * n + col];
            c[row * n + col] = sum;
        //c[row * n + col] = a[row * n + k] * b[k * n + col];  // this cant be used as it will not accumulate the sum across all the same row and column's. Instead it will have only the last row & col element's product.
    }
}



void gemmCPUkernel(int *a, int *b, int *c, int n)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            c[i * n + j] = 0;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                c[i * n + j] += a[i * n + k] * b[k * n + j];
}

int main()
{
    int *d_a, *d_b, *d_c, *d_cT;

    // Host memory
    int *h_a = new int[N * N];
    int *h_b = new int[N * N];
    int *h_c = new int[N * N];
    int *h_cpu = new int[N * N]; 
    int *h_cT = new int[N * N]; // For tiled kernel output
    if (!h_a || !h_b || !h_c || !h_cpu || !h_cT) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return -1;
    }

    // Initialize inputs
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            h_a[i * N + j] = i;
            h_b[i * N + j] = N - i;
            h_c[i * N + j] = 0;
            h_cT[i * N + j] = 0;
            h_cpu[i * N + j] = 0;
        }

    // Allocate GPU memory
    if (cudaMalloc((void**)&d_a, sizeof(int) * N * N) != cudaSuccess ||
        cudaMalloc((void**)&d_b, sizeof(int) * N * N) != cudaSuccess ||
        cudaMalloc((void**)&d_c, sizeof(int) * N * N) != cudaSuccess ||
        cudaMalloc((void**)&d_cT, sizeof(int) * N * N) != cudaSuccess) {
        std::cerr << "CUDA malloc failed!" << std::endl;
        return -1;
    }

    // Copy input to device
    cudaMemcpy(d_a, h_a, sizeof(int) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * N * N, cudaMemcpyHostToDevice);

    //dim3 block(16, 16);
    //Making block size dynamic based on Tile_Width 
    //because our shared memory tiles are also based on TILE_WIDTH*TILE_WIDTH

    // Trying to avoid hardcoding the block size as extra block size - more threads than the available shared memory which is based on TILE_WIDTH*TILE_WIDTH
    

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Timing GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    gemmGPUkernel<<<grid, block>>>(d_a, d_b, d_c, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU execution time: " << milliseconds << " ms\n";

    cudaEventRecord(start);

    gemmGPUTiledkernel<<<grid, block>>>(d_a, d_b, d_cT, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float Tmilliseconds = 0;
    cudaEventElapsedTime(&Tmilliseconds, start, stop);
    std::cout << "Tiled GPU execution time: " << Tmilliseconds << " ms\n";









    // Timing CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    gemmCPUkernel(h_a, h_b, h_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU execution time: " << cpu_duration.count() << " ms\n";

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cT, d_cT, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // Copy result from device to host for tiled kernel


    // Validate result
    bool correct = true;
    for (int i = 0; i < N && correct; ++i){
        for (int j = 0; j < N; ++j) {
            if (h_c[i * N + j] != h_cpu[i * N + j]) {
                std::cerr << "Mismatch at (" << i << "," << j << "): "
                          << "GPU=" << h_c[i * N + j]
                          << ", CPU=" << h_cpu[i * N + j] << "\n";
                correct = false;
                break;
            }

            if (h_cT[i * N + j] != h_cpu[i * N + j]) {
                std::cerr << "Mismatch at (" << i << "," << j << "): "
                          << "GPU=" << h_cT[i * N + j]
                          << ", CPU=" << h_cpu[i * N + j] << "\n";
                correct = false;
                break;
            }    
          }

         }

    std::cout << "Result match: " << (correct ? "YES" : "NO") << std::endl;

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_cT);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_cT;
    delete[] h_cpu;
    cudaDeviceReset();

    return 0;
}
