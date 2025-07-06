---

# 🧠 CUDA 1D Convolution with Shared Memory Optimization

This repository implements and compares **1D convolution kernels in CUDA**, focusing on how **shared memory** and **tiling** techniques impact performance — especially under constraints like **warp divergence** and **predication**.

---

## 📌 Why Convolution?

* **Convolution** is a fundamental operation in signal processing and deep learning.
* It’s ideal for parallelization: **each output element** can be computed independently.
* However, it involves **substantial input overlap** and **challenging boundary conditions**, making it a strong use case for:

  * **Tiling**
  * **Shared memory staging**
  * **Divergence handling**

---

## ⚙️ What is 1D Convolution?

> A **1D convolution** is the **sliding dot product** of a kernel (typically flipped) across an input signal.

### Mathematically:

If input signal `X` has size `n`, and kernel `W` has odd size `2p + 1`:

```
Y[i] = ∑_{j=-p}^{p} X[i + j] * W[j]
```

### Visual Intuition:

![convolution sketch](https://github.com/user-attachments/assets/f5bf9eea-2486-424f-8106-50cd8975bf4e)

Each output element in `Y` is the result of aligning a kernel window around an input point in `X`, with `-p` to `p` coverage. At boundaries, this causes **out-of-bound access** unless handled carefully.

---

## 🧪 Naive Implementation: `oneDConvNaive.cu`

```c
Array A: [0][1][2][3][4][5][6][7][8][9]

For tid = 0, kernel size = 5:
  tries to access: [-2][-1][0][1][2] ❌ OUT OF BOUNDS

For tid = 9, kernel size = 5:
  tries to access: [7][8][9][10][11] ❌ OUT OF BOUNDS
```

This reveals the need to **handle halo regions**, either via:

* Input padding
* Predication
* Zero-padding logic

---

## 🚀 Shared Memory Optimization: `oneDConvTiling.cu`

We implement and compare **two kernels** using shared memory:

---

### 🔻 Kernel 1: `oned_convolution_kernel_warp_partialSHMEMLoad`

#### ❌ Issues:

* Uses **two-step loading**: one for tile, one for halo (conditioned on `offset < n_padded`)
* Causes **warp divergence**, because:

  * Not all threads execute the second memory load
  * Control flow splits within a warp

```cpp
if (offset < n_padded) {
    s_array[offset] = array[g_offset];
}
```

* Final convolution accesses:

  ```cpp
  s_array[threadIdx.x + j] * mask[j]
  ```

  ...which assumes halo is present — dangerous if `offset`-based loading was incomplete.

#### ⚠️ Conclusion:

> Partial shared memory loading leads to warp divergence and potential incorrect behavior in small block sizes.

---

### ✅ Kernel 2: `oned_convolution_kernel_loopUnrolling_lessWarpUsePredication_fullSHMEMLoad`

#### ✅ Improvements:

* Uses a **loop-unrolled, stride-based cooperative loading** approach:

```cpp
for (int i = threadIdx.x; i < blockDim.x + 2*r; i += blockDim.x) {
    s_array[i] = array[blockIdx.x * blockDim.x + i];
}
```

* **All threads follow the same loop structure** → avoids divergence
* Compiler can apply:

  * **Loop unrolling**: fewer instructions
  * **Predication**: threads execute same instructions, only some commit results

#### Example:

* Thread 0 loads: `i = 0, 4, 8`
* Thread 3 loads: `i = 3, 7, 11`
* Even if not all iterations are used, **all threads stay in sync**

#### ✅ Benefits:

* No conditional branching
* High warp execution efficiency
* Fully covers shared memory with no overrun

---

## ✅ When Does Shared Memory Help?

* **Input reuse**: Neighboring threads reuse overlapping data in `s_array[]`
* **Reduces global memory bandwidth**
* Avoids redundant `global loads`

---

## 📚 Summary

| Feature                  | Naive Kernel | Two-Step SHMEM Load       | Loop + Predication Kernel  |
| ------------------------ | ------------ | ------------------------- | -------------------------- |
| Warp Divergence          | ❌ Yes        | ⚠️ Partial (offset logic) | ✅ No (loop stride)         |
| Shared Memory Use        | ❌ None       | ✅ Partial                 | ✅ Full                     |
| Global Memory Efficiency | ❌ Poor       | ✅ Better                  | ✅ Best                     |
| Correctness w/o Padding  | ❌ Unsafe     | ❌ Risky on edges          | ✅ Fully safe               |
| Compiler Optimizations   | ❌ None       | ⚠️ Limited by branches    | ✅ Loop unrolling + masking |

---

## 📦 Files

| File                | Description                              |
| ------------------- | ---------------------------------------- |
| `oneDConvNaive.cu`  | Naive convolution (no shared memory)     |
| `oneDConvTiling.cu` | Shared memory convolution (two versions) |

---

## 💡 Final Notes

* Shared memory is **critical** for high-performance stencil/conv kernels
* **Divergence and halo access** are tightly linked — your approach must ensure:

  * Safe memory access
  * Uniform thread behavior
* Prefer **cooperative tiling with unrolling** when performance and correctness are both priorities

---
