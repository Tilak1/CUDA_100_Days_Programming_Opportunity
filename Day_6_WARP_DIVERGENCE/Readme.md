---

# 🧩 Warp Divergence, SIMD Execution & Shared Memory Optimization

This section documents key concepts from Ch.5 of *Programming Massively Parallel Processors*, extended with real-world CUDA kernel patterns, practical optimization advice, and commentary on **warp behavior**, **divergence**, and **shared memory access patterns**.

---

## 📐 Rectangular Kernels & Shared Memory Access Patterns

### Why?

* Rectangular (tiled) kernels help **maximize data reuse** from global memory, minimizing redundant loads.
* Once data is staged into **shared memory**, it can be reused in multiple threads and reused both row-wise and column-wise.

> **Note:** Shared memory is on-chip and fast; **it doesn't require coalescing** like global memory.

### Common Misunderstanding:

> **Why is row-wise access more expensive than column-wise in linearized global memory?**

Because global memory is linear (row-major), **row-wise accesses by neighboring threads** result in **coalesced memory access**, while **non-strided column accesses** can lead to memory transaction scattering if not handled via tiling.

---

## ⚙️ SIMD Thread Execution and Warp Behavior

* In CUDA, threads are organized into **warps (32 threads)** that execute in lock-step on SIMD hardware.
* **Each warp shares the same threadIdx.x across rows** (in 2D grids), so row-aligned warps are common in stencil and matrix kernels.

---

## ⚠️ Control Divergence in Warps

### What is it?

Control divergence happens when **threads in a warp follow different execution paths**, e.g., due to:

* `if-else` conditions
* `while`/`for` loops that iterate differently per thread

### What happens under the hood?

> Divergence is resolved **sequentially**:

* The hardware runs one control path at a time, **masking inactive threads**.
* Execution time adds up.

### Example:

```cpp
if (threadIdx.x % 2 == 0)
    A[i] += 1;
else
    A[i] -= 1;
```

* Two separate SIMD passes: one for even threads, one for odd.
* Only one group is active at a time; the other is masked.

---

## 📉 Divergence Cost: Example from Ch.5

> The **impact of divergence decreases with problem size**:

* Vector length = **100** → \~25% of threads diverging → high impact
* Vector length = **1000** → \~3% divergence → acceptable
* Vector length = **10,000** → \~0.3% → negligible

---

## 🔁 Divergence in Loops

```cpp
while (val[i] > threshold) {
    ...
}
```

* If threads exit the loop at different times, they will **diverge**
* Loop unrolling/predication won't help unless manually handled

---

## 🧠 Programmer-Controlled Techniques to Avoid Divergence

While compilers help (via **loop unrolling** and **predication**), **you** can do better by writing warp-uniform code.

---

### 🔧 1. Use Warp-Uniform Conditions

Instead of:

```cpp
if (threadIdx.x < 5) { ... }
```

Use:

```cpp
bool active = (threadIdx.x < 5);
float value = active ? compute() : 0.0f;
```

🧠 **Why?**
Now all threads execute the same instructions — divergence is avoided, and control flow is uniform.

---

### 🔁 2. Use Loop Striding Instead of Fixed-Range Threads

Bad:

```cpp
if (threadIdx.x < N) {
    C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}
```

Better:

```cpp
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    C[i] = A[i] + B[i];
}
```

✅ All threads execute same loop → no divergence.

---

### 🚫 3. Avoid Divergence Inside Loops

Bad:

```cpp
while (val[i] > threshold) {
    ...
}
```

Better:

```cpp
int iter = 0;
while (iter < MAX_ITERS) {
    bool active = (val[i] > threshold);
    if (!active) break;
    ...
    iter++;
}
```

Or use masks:

```cpp
for (int i = 0; i < MAX; i++) {
    if (condition) {
        result += A[i];
    }
}
```

---

### ⚙️ 4. Manual Predication Instead of Conditionals

```cpp
s_array[i] = (in_bounds ? array[i] : 0);
```

Better still:

```cpp
s_array[i] = __any_sync(0xFFFFFFFF, in_bounds) ? array[i] : 0;
```

✅ Warp-uniform masked memory access without branching.

---

### 🧩 5. Thread Compaction and Bitmasking (Advanced)

For sparse activations or conditional work:

```cpp
unsigned mask = __ballot_sync(0xFFFFFFFF, active);
int leader = __ffs(mask) - 1;

if (threadIdx.x == leader) {
    do_work();
}
```

Use with:

* `__any_sync`
* `__ballot_sync`
* `__ffs`, `__popc`

This enables warp-efficient subgroup work execution.

---

## ✅ Summary Table

| Goal                           | Technique                                  |
| ------------------------------ | ------------------------------------------ |
| Uniform thread control         | Use warp-uniform `if` and loop conditions  |
| Reduce divergence inside loops | Use predication / loop guards              |
| Avoid control flow imbalance   | Loop striding, cooperative tiling          |
| Avoid memory divergence        | Use padding and shared memory              |
| Explicit warp control          | Use `__ballot_sync`, `__any_sync`, masking |

---

## 📚 Reference

> Chapter 5 - *Performance Considerations*,
> [Programming Massively Parallel Processors - A Hands-on Approach (3rd Ed)](https://github.com/R100001/Programming-Massively-Parallel-Processors/tree/master/Chapters/Ch05%20-%20Performance%20Considerations)

---
