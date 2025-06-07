Rectangular Kernels: 

For an even more reduced global memory usage. 

5.1, 5.2 not understood - Need to redo again


Till now we have seen memory colaescing techniques and tiling (for optimized gloabl memory access and barrier synchronization) to improve the gemm performance by improving communication and computation tradeoff.

PS: Once the data is in shared memory, they can be accessed either on a row basis or a column basis with much less performance variation because the shared memories are implemented as intrinsically high-speed on-chip memory that does not require coalescing to achieve high data access rate.

Why accessing rows takes more toll than a column in a linearized memory access ? 
---
Threads execution in SIMD: 

Threads in 2D are laid out just similar to the row major order. All the threadsin the same warp have the same threadID.x

Threads in the same WARP follow their own path or allow to be divergent - like an if else condition - they are free and will not be stopped or let the execution time add up. 



Control divergence issue and reduction ? 

Ch-5: 

Warps and SIMD Hardware: 

The SIMD hardware will take multiple passes through these divergent paths. One pass executes those threads that follow the if part and another pass executes those that follow the else part. During each pass, the threads that follow the other path are not allowed to take effect. These passes are sequential to each other, thus will add to the execution time.

While the hardware executes the same instruction for all threads in a warp, it selectively lets the threads take effect in only each pass, allowing every thread to take its own control flow path. This preserves the independence of threads while taking advantage of the reduced cost of SIMD hardware

Divergence also can arise in other constructs, for example, if threads in a warp execute a for-loop which can iterate six, seven, or eight times for different threads. All threads will finish the first six iterations together. Two passes will be used to execute the 7th iteration, one for those that take the iteration and one for those that do not. Two passes will be used to execute the 8th iteration, one for those that take the iteration and one for those that do not.

Author gives the 2 out of 8 analogy for what might be affected here: 

Note that the performance impact of control divergence decreases with the size of the vectors being processed. For a vector length of 100, one of the four warps will have control divergence, which can have significant impact on performance. For a vector size of 1000, only one out of the 32 warps will have control divergence. That is, control divergence will affect only about 3% of the execution time. Even if it doubles the execution time of the warp, the net impact to the total execution time will be about 3%. Obviously, if the vector length is 10,000 or more, only one of the 313 warps will have control divergence. The impact of control divergence will be much less than 1%!

Using reduction algorithms on warp & thread optimization: 

The difference between the two kernels is small but has very significant performance impact. It requires someone with clear understanding of the execution of threads on the SIMD hardware of the device to be able to confidently make such adjustments.

The automatic variables declared in a CUDA kernel are placed into registers. By dynamically partitioning the registers among blocks, the SM can accommodate more blocks if they require few registers, and fewer blocks if they require more registers.



---
Ref: 

https://github.com/R100001/Programming-Massively-Parallel-Processors/tree/master/Chapters/Ch05%20-%20Performance%20Considerations
