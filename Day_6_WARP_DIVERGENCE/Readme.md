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

🔧 1. Use Warp-Uniform Conditions
Ensure all threads in a warp take the same path.

Instead of:

cpp
Copy
Edit
if (threadIdx.x < 5) { ... }
Use:

cpp
Copy
Edit
bool enable = (threadIdx.x < 5);
int value = enable ? compute() : 0;
Or better — redesign so that all threads do the same work, or shift logic outside divergence-critical paths (e.g., inside if (blockIdx.x == 0) which is warp-uniform).

🔁 2. Loop Striding Instead of Conditionals
Instead of:

cpp
Copy
Edit
if (threadIdx.x < N) {
    out[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}
Use:

cpp
Copy
Edit
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    out[i] = A[i] + B[i];
}
✅ No divergence: all threads loop over same structure

🚫 3. Avoid Divergence Inside Loops or Kernels
Divergence inside loops (especially with break, continue, return) is the most expensive.

Instead of:

cpp
Copy
Edit
while (val[i] > threshold) {
    ...
}
Try to use:

cpp
Copy
Edit
int iter = 0;
while (iter < MAX_ITERS) {
    bool active = (val[i] > threshold);
    if (!active) break;
    ...
    iter++;
}
Or use flags to mask computation:

cpp
Copy
Edit
for (int i = 0; i < MAX; i++) {
    bool active = (condition);
    compute only if active;
}
⚙️ 4. Use Predicated Instructions Manually
Replace:

cpp
Copy
Edit
if (in_bounds) {
    s_array[i] = array[i];
}
With:

cpp
Copy
Edit
s_array[i] = in_bounds ? array[i] : 0;
Or even better:

cpp
Copy
Edit
s_array[i] = __any_sync(0xFFFFFFFF, in_bounds) ? array[i] : 0;
This makes the logic uniform across threads and relies less on compiler branching logic.

🧩 5. Thread Compaction and Bitmasking (Advanced)
If only a subset of threads need to do something:

Let one warp compact the work into a smaller subgroup

Use warp-level intrinsics:

cpp
Copy
Edit
unsigned mask = __ballot_sync(0xFFFFFFFF, active);
int leader = __ffs(mask) - 1;
Then only the leader thread (or a subset) executes something.

This is the basis of thread compaction, warp voting, and coalesced divergence mitigation.

🧠 Summary Table
Goal	How to Avoid Divergence
Ensure all threads follow same path	Use uniform if conditions, based on warp-wide vars
Avoid short divergent paths	Convert if to ? : ternary / mask compute
Avoid divergent loops	Use loop striding / predicated for loops
Avoid divergence in memory access	Pad memory and load cooperatively
Control warp behavior explicitly	Use warp intrinsics: __any_sync, __ballot_sync



---
Ref: 

https://github.com/R100001/Programming-Massively-Parallel-Processors/tree/master/Chapters/Ch05%20-%20Performance%20Considerations
