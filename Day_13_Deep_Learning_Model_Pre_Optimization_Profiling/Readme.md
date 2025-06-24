# 🧠 Day 13: Optimizing a DL Model

## 🚫 Why GPUs Not Go Brrr?

Despite having massive compute potential, GPUs often **don’t go brrr** due to:

- **Memory Bottlenecks** – Limiting throughput due to excessive global memory access.
- **Overhead** – Time wasted in non-compute activities (Python interpreter, kernel launch delays, etc.).

Our goal: ✅ Relieve these bottlenecks and make sure **GPUs go brrr** 💨.

---

## 🧵 Memory Bound?

To reduce global memory accesses, **operator fusion** is key. This combines multiple operations into one GPU kernel, minimizing memory I/O.

### ✅ Operator Fusion Example:
![Operator Fusion](https://github.com/user-attachments/assets/d19c1dce-6869-4d5c-b260-62fea339ebbd)

### How?

- **PyTorch Eager Mode** already applies some fusion.
- **Compilers like `nvFuser` and `XLA`** can do more advanced fusions.
- But for peak performance, **handwriting custom CUDA kernels** is king 👑.

> 🎯 Motivation unlocked: Time to learn GPU kernels! 🔥

---

## 🔢 Compute Bound?

One way to identify if you're compute-bound:
- Measure **achieved FLOPS** as a % of your GPU’s **peak theoretical FLOPS**.

If this number is low, you're either:
- Limited by memory access (again), or
- Not launching enough parallel work to saturate GPU compute resources.

---

## ⚙️ Overhead?

Overhead = Time spent **not doing actual compute or tensor transfers**.

Examples:
- Python interpreter time 🐍
- PyTorch framework internal time 🧱
- CUDA kernel *launch* delays (not execution!) 🚀

### Visualization:
![Overhead Breakdown](https://github.com/user-attachments/assets/e288d349-5d27-4754-8b82-dd7031096b75)

---

## 🔍 Application: Profiling U-Net

Let’s now apply all the concepts above and **profile a U-Net** model:
- Identify memory bottlenecks
- Look at kernel launch vs execution
- Measure utilization
- Use fusion and/or custom kernels where needed

---

## 📚 References

- 📝 [Horace.io: GPU Go Brrr!](https://horace.io/brrr_intro.html)  
- 💡 [Inference Optimization Blog Series](https://github.com/vdesai2014/inference-optimization-blog-post/tree/main/part-3)
