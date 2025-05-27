
---

# CUDA Thread Block and Grid Configuration

## 📌 One-Dimensional Data

When working with 1D data, correctly defining the number of **blocks** after specifying the number of **threads per block** is crucial.

### 🔢 Computing `numBlocks`

* **Floating-point approach** (conceptual):

  ```cpp
  numBlocks = ceil(length / numThreadsPerBlock);
  ```

  This uses floating-point math but returns a non-integer type.

* ✅ **Integer-friendly trick** (preferred):

  ```cpp
  numBlocks = (length + numThreadsPerBlock - 1) / numThreadsPerBlock;
  ```

  This effectively computes the ceiling without needing floating-point operations.

### ✅ Example

```cpp
int length = 1000;
int numThreadsPerBlock = 256;

int numBlocks = (length + numThreadsPerBlock - 1) / numThreadsPerBlock;  // = 4
```

* `256 * 3 = 768` → Not enough
* `256 * 4 = 1024` → Just enough (overshoot is OK)

### 🛡️ Overshoot Logic

Always guard your kernel code:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < length) {
    // Safe to access data[i]
}
```
