#include <iostream>
#include <cstdlib> // for rand()
#include <ctime> // for time() - srand(time(0))


#include <cuda_runtime.h>
#include <stdio.h>
#include "timer.h"

// Since RBG values - 0-255 - All in Char Data type

__global__ void rgb2gray_kernel(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) { // cond for last block's unsued thread condition
        int idx = row * width + col; // making a 1D index as the RBG are 1D char arrays 
        // index in a 2D image = row * width + col
        // Every thread will have a diff index as theu have a diff row and col 
        gray[idx] = (unsigned char)(0.299f * red[idx] + 0.587f * green[idx] + 0.114f * blue[idx]);
    }
}

void rgb2gray_gpu(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height) {
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    unsigned char *red_d, *green_d, *blue_d, *gray_d;
    cudaMalloc((void**)&red_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&green_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&blue_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&gray_d, width * height * sizeof(unsigned char));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(red_d, red, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);
    // rgb2gray_kernel<<<gridSize, blockSize>>>(red_d, green_d, blue_d, gray_d, width, height);    
    
    // Considering a 32x32 image - 4 blocks = each block is 
    // 4 blocks of each 4x4
    // Appplying the same numBlocks formula from one dimension in Day 2 to multi dimenisonal 
    dim3 numThreadPerBlock(32, 32); // 16x16x1 1 in z is a default value
    dim3 numBlocks(width+numThreadPerBlock.x-1/numThreadPerBlock.x, height+numThreadPerBlock.y-1/numThreadPerBlock.y, 1);
    rgb2gray_kernel<<<numBlocks,numThreadPerBlock>>>(red_d, green_d, blue_d, gray_d, width, height);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel execution time");

    // Copy result back to host
    startTime(&timer);
    cudaMemcpy(gray, gray_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy back to host time");

    // Free memory
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);
}



int main(){

int width = 32;
int height = 32;
unsigned char *red = (unsigned char*)malloc(width * height * sizeof(unsigned char));
unsigned char *green = (unsigned char*)malloc(width * height * sizeof(unsigned char));
unsigned char *blue = (unsigned char*)malloc(width * height * sizeof(unsigned char));
unsigned char *gray = (unsigned char*)malloc(width * height * sizeof(unsigned char));
// Initialize the RGB arrays with random values
srand(time(0)); // Seed for random number generation
for (int i = 0; i < width * height; ++i) {
    red[i] = rand() % 256;
    green[i] = rand() % 256;
    blue[i] = rand() % 256;
}

// Call the GPU function
rgb2gray_gpu(red, green, blue, gray, width, height);

// Print the grayscale image (for demonstration purposes)
for (int i = 0; i < width * height; ++i) {
    printf("%d ", gray[i]);
    if ((i + 1) % width == 0) {
        printf("\n");
    }  


}
}