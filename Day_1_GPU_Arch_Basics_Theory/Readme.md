
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

## 🔍 **Memory Access on GPUs**

* **Global memory (DRAM)** access is **high-latency**.
* If threads frequently access global memory without optimization, **memory stalls** cause **compute under-utilization**.
* 🔧 **Best Practices**:

  * Use **shared memory**, **constant memory**, and **registers** for faster access.

![Memory Hierarchy](https://github.com/user-attachments/assets/fdc5aba5-849c-47ad-bafc-385d0069f9fa)

---

## 🧱 **GPU Architecture Highlights**

* **Smaller caches (L1/L2)** compared to CPUs.
* **Many-core architecture**: GPUs use many **lightweight cores**, unlike the fewer powerful CPU cores.
* Threads are grouped into **blocks**, which are assigned to **Streaming Multiprocessors (SMs)**.
* SMs schedule and execute **blocks concurrently or sequentially**, depending on resource availability.
* More **SMs = Higher throughput & parallelism**.

---

## 💻 **CUDA Overview (Intro)**

> *(Will be expanded in later chapters)*

* CUDA allows writing **GPU kernels** in C/C++ for NVIDIA GPUs.
* Key abstractions:

  * **Threads per block**
  * **Blocks per grid**
* CUDA exposes hierarchical **thread and memory models** for fine-grained optimization.

---

### 📚 **References (Day 1)**

* 📖 [PMPP GitHub Book Chapter 1](https://github.com/R100001/Programming-Massively-Parallel-Processors/tree/master/Chapters/Ch01%20-%20Introduction)
* 📘 [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance-guidelines)
* 🧠 [Intro to GPUs – Diffusion Policy Blog](https://www.vrushankdes.ai/diffusion-policy-inference-optimization/part-i---intro-to-gpus)

---

Let me know if you'd like a printable or OneNote-friendly version, or if you're working on Day 2 and want a similar format.
