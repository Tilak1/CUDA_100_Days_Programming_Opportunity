Day 4 Notes: 
---

# GPU Architecture and Matrix Multiplication Optimization

Modern GPUs combine both Harvard and Von Neumann architecture principles—Harvard in the sense of having separate caches for instructions and data, and Von Neumann by allowing unified memory access at the global level. This hybrid architecture is covered well in Stanford's CS336: LLM from Scratch.

In CUDA programming, memory hierarchy is central to performance. There are several types of GPU memory. Global memory is large but slow, shared memory is smaller and faster, and registers are the fastest but extremely limited. Here are some illustrations:

![GPU Memory Types 1](https://github.com/user-attachments/assets/6004d111-fe63-4aa8-ac8b-5a60df7854a3)  
![GPU Memory Types 2](https://github.com/user-attachments/assets/64604cbf-5503-468e-8d1a-d780b75e87b1)  
![GPU Memory Types 3](https://github.com/user-attachments/assets/7ff924b7-79b4-4b29-98af-156ad186442b)

Global memory has high latency and limited bandwidth, whereas shared memory—though faster—still requires explicit load instructions. Registers, by contrast, are accessed almost instantly. In modern GPUs with ~1000 GB/s memory bandwidth, only about 250 GFLOPs/s are achievable via global memory reads (assuming 4 bytes per float), unless reuse is exploited.

In matrix multiplication, the kernel typically looks like this:

```cpp
for (int k = 0; k < Width; ++k)
    Pvalue += M[Row * Width + k] * N[k * Width + Col];
````

This accesses M and N from global memory each iteration and performs a single multiply-add (1 FLOP). The compute-to-global-memory-access ratio here is 1.0, meaning each FLOP requires two memory accesses. This is extremely inefficient and can result in less than 2% of peak GPU throughput. To improve, we must raise this ratio by an order of magnitude—i.e., perform 10+ FLOPs per memory access.

Here’s the layout:

![Matrix Access Logic](https://github.com/user-attachments/assets/7baef47f-5ca7-4976-a896-627935b48f2f)

The sentence, *"We need to increase the ratio by at least an order of magnitude for the computation throughput of modern devices to achieve good utilization,"* means that GPUs need more computation relative to the memory they access. Memory bandwidth is the bottleneck, not compute capability. Hence, to keep GPU ALUs busy, we must reuse data—using shared memory, registers, or caches—rather than repeatedly fetching from global memory.

This is where shared memory tiling becomes critical. Threads within a CUDA block can synchronize, pre-load a tile of data into shared memory, and operate on it collaboratively. This avoids redundant loads and reduces memory access latency significantly. A good visualization of this tiling and thread collaboration strategy looks like this:

![Threads Collab Memory Access](https://github.com/user-attachments/assets/79030443-fc2a-4431-902c-e9ef257eaecf)

Frameworks like Triton, used in the [Caterpillar Project](https://github.com/yogeshsinghrbt/caterpillar), automate many of these optimizations. Tiling, shared memory reuse, and blocking help increase arithmetic intensity—more computation per memory access—which is essential for achieving high throughput in matrix-heavy GPU applications like deep learning or baseband signal processing.

Additional references:

* [Stanford CS336: LLM from Scratch](https://web.stanford.edu/class/cs336/)
* [Programming Massively Parallel Processors - Chapter 4](https://github.com/R100001/Programming-Massively-Parallel-Processors/tree/master/Chapters/Ch04%20-%20Memory%20And%20Data%20Locality)
* [Caterpillar Project using Triton](https://github.com/yogeshsinghrbt/caterpillar)
* [YouTube Talk on GPU Memory Access](https://www.youtube.com/watch?v=6OBtO9niT00)

Optimize for reuse, minimize global memory loads, and match GPU compute intensity to memory bandwidth to fully utilize modern architectures.

```
