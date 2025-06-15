
---

# 📘 **PMPP Book – Chapter 1 Summary: Introduction to Massively Parallel Processors**

## 🚀 **CPU vs GPU**

![CPU vs GPU](https://github.com/user-attachments/assets/8ef5c315-4310-4a7a-8e19-3e90b1dbee99)

* **CPU**: Designed for **serial execution**, with a few **high-performance cores**.
* **GPU**: Optimized for **massive parallelism**, featuring **hundreds to thousands** of lightweight cores.

> ✅ GPUs excel when a **large portion of a workload** is **parallelizable**, especially with **data parallelism** (same operation on different data).

![Data Parallelism](https://github.com/user-attachments/assets/fd296e50-5974-497a-ab0c-616ddfae5396)

### ❓ Why do GPUs have less L1/L3 cache?

![Cache comparison](https://github.com/user-attachments/assets/852a8bb9-62ea-4432-b97b-d0a6870c0be3)

* GPUs **tolerate long-latency memory operations** through **massive thread-level parallelism**, not large caches.
* This allows more silicon to be allocated to **floating-point execution units** rather than caches or branch predictors.
* More on this in Day 3 (Warp scheduling & latency hiding).

---

## 🧱 **GPU Architecture Highlights**

* **Smaller caches (L1/L2)** compared to CPUs.
* **Many-core architecture**: GPUs use many **lightweight cores**, unlike the fewer powerful CPU cores.
* Threads are grouped into **blocks**, which are assigned to **Streaming Multiprocessors (SMs)**.
* SMs schedule and execute **blocks concurrently or sequentially**, depending on resource availability.
* More **SMs = Higher throughput & parallelism**.

![Memory Hierarchy](https://github.com/user-attachments/assets/fdc5aba5-849c-47ad-bafc-385d0069f9fa)

---

## **Shared Memory:**

* Each SHM bank does have a limit of serving 1 byte per clock cycle. As a result, it is most efficient to have each thread access **sequential shared memory addresses** to prevent accesses from becoming serialized. It should be noted that this latency penalty arises due to resource contention between threads trying to access the same memory pool and not due to the ordering of memory addresses within the same pool.

![image](https://github.com/user-attachments/assets/455c8625-2a04-408b-a8b2-254fad61c5ac)

* In the example above, the access pattern on the left and right are bank-conflict free. The pattern in the middle however, involves multiple threads simultaneously accessing the same SRAM bank. The SRAM memory controller in this case would have to serialize these accesses and the net result would be the access being twice as slow.
* Shared memory can also be useful for coalescing a strided global memory load, which if performed directly into thread registers may have required very slow un-coalesced access.
* Overusing SMEM will prevent the warp scheduler from making the SM fully occupied and hurt overall kernel performance.
---
## Thread Regsters: 

* On an RTX 3090, threads cannot use more than 255 registers without spilling to ‘local memory’ which is somewhat deceptively named as it refers to an address space that threads can utilize in global memory.
---
## Warps-Register Spilling: 

* Often times warps will utilize a large number of registers and then ‘stall’ due to memory dependencies. When the cumulative size of all register files for issued warps exceeds the physical register space on the SM, this can lead to ‘register spilling’. In these cases the GPU will fall back to storing local variables in L2 cache or global memory. The key takeaway here is that the GPU is designed to very quickly swap in different workloads to hide memory latency (define this on the side). So long as there are enough warps to throw into the fray, memory loading and compute can overlap. It’s the programmers job to understand this concept (occupancy) and make sure the GPUs compute execution units can keep the hungry compute units fed.

---

## Tensor Cores: 

* Each tensor core can execute an entire 16x16 mma (matrix multiply and accumulate) in a single clock cycle (~0.75ns for RTX 3090).
![image](https://github.com/user-attachments/assets/daf483c7-ff07-4228-af85-982868f36809)

* The operation is performed as a warp-wide operation as all 32 threads have to participate in loading the input matrices into the tensor cores, which breaks the thread/warp abstraction to some extent in CUDA.



## 🔍 **Memory Access on GPUs**

* **Global memory (DRAM)** access is **high-latency**.
* If threads frequently access global memory without optimization, **memory stalls** cause **compute under-utilization**.
* 🔧 **Best Practices**:
  * Use **shared memory**, **constant memory**, and **registers** for faster access.


![image](https://github.com/user-attachments/assets/2faea4c7-aae3-4b0b-8798-414ea12e33e1)

    
Efficient use of shared memory/registers and coalesced global memory access can unlock high memory throughput. This in turns allows you to keep the thousands of cores on-chip fed. A kernel with healthy memory usage practices and effective overlapping of memory access with compute will go very far in achieving high overall hardware utilization.

![Alt text](https://cdn.prod.website-files.com/65cbfd86576f83a5d9e2875e/65ea63fcd777bf1e209e71f2_gpu_memory_hierarchy_rev2.gif)


---

## 💻 **CUDA Overview (Intro)**

> *(Will be expanded in later chapters)*

* CUDA allows writing **GPU kernels** in C/C++ for NVIDIA GPUs.
* Key abstractions:

  * **Threads per block**
  * **Blocks per grid**
* CUDA exposes hierarchical **thread and memory models** for fine-grained optimization.

![image](https://github.com/user-attachments/assets/be7da3a2-0239-4355-b675-6db66457b64d)


---

### 📚 **References (Day 1)**

* 📖 [PMPP GitHub Book Chapter 1](https://github.com/R100001/Programming-Massively-Parallel-Processors/tree/master/Chapters/Ch01%20-%20Introduction)
* 📘 [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance-guidelines)
* 🧠 [Intro to GPUs – Diffusion Policy Blog](https://www.vrushankdes.ai/diffusion-policy-inference-optimization/part-i---intro-to-gpus)

---
