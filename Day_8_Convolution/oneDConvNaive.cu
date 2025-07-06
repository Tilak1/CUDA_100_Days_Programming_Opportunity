#include <iostream>
#include <cuda_runtime.h>
#define Mask_width 5  
__constant__ float M[Mask_width];


__global__ void oned_convolution_kernel(const float* A, float* C, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float result = 0.0f;
        int r = Mask_width / 2;
        int start = tid - r;

        for (int j = 0; j < Mask_width; j++) {
            if ((start + j) >= 0 && (start + j) < n) {
                result += A[start + j] * M[j];
            }
        }
        
        C[tid] = result;
    }
}


// Host function to check for CUDA errors
void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << message << " - CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void verify_result(int* A, int *M int * C, int n, int m)
{

int r = m/2; 
int result[n];
for (int i = 0; i < n; i++) {
    result[i] = 0;
    int start = i - r;

    for (int j = 0; j < m; j++) {
        if ((start + j) >= 0 && (start + j) < n) {
            result[i] += A[start + j] * M[j];
        }
    }

assett(temp == result[i]); 


} 

    for (int i = 0; i < n; i++) {
        if (C[i] != result[i]) {
            std::cerr << "Verification failed at index " << i << ": expected " << result[i] << ", got " << C[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "Verification passed!" << std::endl;
}



int main(){
 
  int n=10;
  float A[n],C[n];
  float d_M[Mask_width];
  
   for (int i=0; i<Mask_width;i++){
    d_M[i]=i;

  }
  for (int i=0; i<n;i++){
    A[i]=i;

  }

  float *d_a,*d_c;

  cudaMalloc(&d_a,n*sizeof(float));
  cudaMalloc(&d_c,n*sizeof(float));
  cudaMemcpy(d_a,A,n*sizeof(float),cudaMemcpyHostToDevice);
  checkCudaError("Failed to copy input data to device");
  cudaMemcpyToSymbol(M,d_M,Mask_width*sizeof(float));
  checkCudaError("Failed to copy mask data to device");

  dim3 dimBlock(32);
  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
  
  oned_convolution_kernel<<<dimGrid, dimBlock>>>(d_a,d_c,n);// not passing the M coz its a GPU const memory - already there 
  checkCudaError("Failed to execute the kernel");
  cudaDeviceSynchronize();
  cudaMemcpy(C,d_c,n*sizeof(float),cudaMemcpyDeviceToHost);
  checkCudaError("Failed to copy output data to host");
  cudaFree(d_a);
  cudaFree(d_c);
  
  printf("GPU results !\n");

  //printing the results
  printf("A:\n");
  for (int i=0; i<n;i++){
    printf("%.2f ", A[i]);

  }
  printf("\n");
   printf("\nd_m:\n");
    for (int i = 0; i < Mask_width; i++) {

            printf("%.2f ", d_M[i]);

    }
  printf("\n");
  printf("\nC:\n");
    for (int i = 0; i < n; i++) {

            printf("%.2f ", C[i]);

    }

    printf("\n");
    // Verify the result
    printf("\nVerifying the CPU and GPU result...\n");  
  verify_result(A, d_M, C, n, Mask_width);

  printf("\n");
}