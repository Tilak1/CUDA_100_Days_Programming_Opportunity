Here's a well-formatted and structured `README.md` based on your notes from Chapter 1 of the PMPP book:

---

# 📘 PMPP Book – Chapter 1 Summary

## 🚀 CPU vs GPU

![image](https://github.com/user-attachments/assets/8ef5c315-4310-4a7a-8e19-3e90b1dbee99)

* **CPU**: Optimized for **serial execution** with few, high-performance cores.
* **GPU**: Designed for **massive parallel execution**, using hundreds to thousands of smaller cores.

> ✅ Using a GPU is ideal **only when** a large portion of the workload can be executed in parallel. Same operations on different data – so can parallelize / comes into the Data prallelism. 
![image](https://github.com/user-attachments/assets/fd296e50-5974-497a-ab0c-616ddfae5396)

> Why GPU's have less cache and L1/L3
![image](https://github.com/user-attachments/assets/852a8bb9-62ea-4432-b97b-d0a6870c0be3)
* Ability to tolerate long-latency operations is the main reason GPUs do not dedicate nearly as much chip area to cache memories and branch prediction mechanisms as do CPUs. Thus, GPUs can dedicate more of its chip area to floating-point execution resources.
* This will be discussed more on Day 3 notes (Thread scheduling - WARP - latency)


---

## 🔍 Memory Access on GPUs

* **Global memory (DRAM)** access is **costly** and slow.
* If a kernel frequently accesses global memory, **memory latency dominates** and the **compute cores sit idle**.
* **Best practice**:
  Use **shared memory**, **constant memory**, and **registers** wherever possible to reduce latency.

---

## 🧱 GPU Architecture Highlights

* GPUs have **limited L1/L2 cache** compared to CPUs.
* **Many-core architecture**: Unlike CPUs which have few powerful cores, GPUs have **many lightweight cores**.
* Threads are organized into **blocks**.
* **Blocks are assigned to SMs (Streaming Multiprocessors)** for execution:

  * Execution may be **sequential or concurrent** depending on scheduling.
* GPUs with **more SMs** generally have **higher throughput** and performance.

---

## 💻 CUDA Overview

> *(To be expanded in next chapters)*

* CUDA enables writing C/C++-like code for **NVIDIA GPUs**.
* Programmer specifies:

  * Number of **threads per block**
  * Number of **blocks per grid**
* CUDA exposes fine-grained control over thread and memory hierarchies.

---

Ref for Day 1: 

https://github.com/R100001/Programming-Massively-Parallel-Processors/tree/master/Chapters/Ch01%20-%20Introduction

https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance-guidelines
