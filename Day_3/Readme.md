**One Dimensional Data:**

* After defining the numThreads, we need to correctly define the numBlocks.  
* Floating point world: numBlocks = ceil(length of data / numThreadsPerBlock). But ceil gives floating point. 
  
* Integer friendly trick: numBlocks = (length + numThreadsPerBlock - 1) / numThreadsPerBlock - will give an integer value similar to ceil / This gives the smallest integer greater than or equal to length / numThreadsPerBlock.

    **Example:**
 
    int length = 1000;
    int numThreadsPerBlock = 256;
    
    int numBlocks = (length + numThreadsPerBlock - 1) / numThreadsPerBlock;  // = 4

    256 * 3 = 768 → not enough
    256 * 4 = 1024 → just enough (overshoot is OK, you just guard with if (i < length))

    **1D Overshoot logic:** 
      if < length
     
**Multi Dimensional Data:**

In a 2D dimension, we can write: 
  dim3 numBlocks((width  + threadsPerBlock.x - 1) / threadsPerBlock.x,
               (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

   **2D Overshoot logic:** 
   unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
   unsigned int col = blockIdx.x * blockDim.x + threadIdx.x; 
 
   if (row < height && col < width) { // cond for last block's unsued thread condition
 
**Concept of flexible resource assignment and the concept of occupancy:**


