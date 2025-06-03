Till now we have seen memory colaescing techniques and tiling (for optimized gloabl memory access and barrier synchronization) to improve the gemm performance by improving communication and computation tradeoff.

PS: Once the data is in shared memory, they can be accessed either on a row basis or a column basis with much less performance variation because the shared memories are implemented as intrinsically high-speed on-chip memory that does not require coalescing to achieve high data access rate.

Why accessing rows takes more toll than a column in a linearized memory access ? 


Ref: 

https://github.com/R100001/Programming-Massively-Parallel-Processors/tree/master/Chapters/Ch05%20-%20Performance%20Considerations
