### CUDA Streams: 

* Need to launch and use the GPU asynchronously without waiting for any dependency (i.e memory - aka H2D & D2H copy) and experience the true parallel independent arithmetic GPU operations (i.e compute) ? - CUDA streams is the asnwer to evade this serial approach. 

Default Stream: Essentially when we call cudaMemcpy or we do not specify stream when we call cudaMemcpyAsync, we are using the default stream.

![image](https://github.com/user-attachments/assets/702ac27e-b4ed-4812-ac56-d76c604a599f)

We can see the time saved in the above image

Requirements: 
* Need the Host pinned memory to do Async host operation. This host's page-locked memory is the same as normal heap memory (that we would allocate through malloc), except it is guaranteed never to be swapped into the virtual memory on disk by the OS. OS makes sure that the memory is always within the physical memory of the system.

  *   checkCuda( cudaMallocHost((void**)&a, bytes) );      // host pinned
  *   cudaHostAlloc( (void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault );

* GPU can use direct memory access to copy data from this location in the host.
  
  *   Up until now, we have used the cudaMemcpy. This is a synchronous function that only returns when the copy is completed.
  *   We will use cudaMemcpyAsync which returns to the caller immediately. The copy operation is completed at some time after this call.

Aftet this asynchronous call of memcpyAsync, Kernel will also be called aysncly 

## CUDA Multi Streams:  

## CUDA Multi Device Streams; 


## Using Events with streams 



## Ref: 
https://leimao.github.io/blog/CUDA-Stream/

https://turing.une.edu.au/~cosc330/lectures/display_notes.php?lecture=22

https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf

https://turing.une.edu.au/~cosc330/lectures/display_notes.php?lecture=22

