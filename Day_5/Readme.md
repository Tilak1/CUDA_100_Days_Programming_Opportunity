

## 📅 Day 5 Notes: Matrix Multiplication with CUDA

### Basic GEMM (gemm.cu) metrics:

Matrix size `N x N` wqdq

```
N = 100
GPU execution time:   22.1696 ms  
CPU execution time:    4.57534 ms  

N = 1000
GPU execution time:   22.2881 ms  
CPU execution time: 4902.46 ms  
```

---

### ❓ Question:

Can tiled implementation **beat** the above metrics?

---

## 🚀 Tiled GEMM Implementation:

### 🔍 Motivation:

* Use of **shared memory** via `__shared__` and **barrier synchronization** using `__syncthreads()` to avoid repeated global memory fetches.
* Load a block-sized tile of matrices **A** and **B** into shared memory **once**, and allow all threads in the block to compute on this tile.
* This saves time as **shared memory is much faster than global memory**.
* The logic ensures we **reuse** portions of matrices A and B.

---

### 🧱 Core Logic Breakdown:

* Each block handles a **tile** of the result matrix.
* Each thread calculates a single element `c[row][col]` by looping over shared memory tiles.

```cpp
Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
```

---

### 📦 Shared Memory Loading:

```cpp
__shared__ int T_a[TILE_WIDTH][TILE_WIDTH];
__shared__ int T_b[TILE_WIDTH][TILE_WIDTH];
```

* Threads in the block **collaboratively** load a tile of A and B:

```cpp
T_a[ty][tx] = a[row * n + (ph * TILE_WIDTH + tx)];
T_b[ty][tx] = b[(ph * TILE_WIDTH + ty) * n + col];
```

* This is a **tile-phase-based access**:

  * `ph` is the **phase number** of the tile
  * `TILE_WIDTH * TILE_WIDTH` shared memory usage must **fit into GPU’s shared memory**

---

### 🔁 Synchronization:

```cpp
__syncthreads();
```

Used twice:

1. After loading tiles
2. After performing multiply-adds before moving to the next tile

---

### 🧮 Final Computation:

```cpp
for (int k = 0; k < TILE_WIDTH; ++k)
    sum += T_a[ty][k] * T_b[k][tx];
```

---

## 🧠 Notes on Thread-Index Mapping:

* In shared memory:

  * `ty` → row index within tile
  * `tx` → col index within tile
    Hence shared memory loading is:

```cpp
T_a[ty][tx]   // matches A[row][tile_col]
T_b[ty][tx]   // matches B[tile_row][col]
```

---

## ✅ Dynamic Block Size:

Avoid hardcoding:

```cpp
dim3 block(TILE_WIDTH, TILE_WIDTH);
dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
```

This was done to have the same block size as the tiles. 

If blockDim != TILE_WIDTH, then:

Either some threads write out-of-bounds, or

Some slots are never filled, leading to wrong results.

---

### 🔬 Numerical Indexing Example:

* Suppose `N = 4`, `TILE_WIDTH = 2`
* `grid` = (2, 2), `block` = (2, 2)
* Threads (0,0) to (1,1) in a block will:

  * Load subtiles `T_a[ty][tx]` and `T_b[ty][tx]` for current phase `ph`
  * Compute and accumulate `sum`
  * Store `c[row][col] = sum`

---

## 📊 Tiled GEMM metrics (gemmTiled.cu):

```
N = 256
Tile_Width = 2
GPU execution time:          31.1038 ms  
Tiled GPU execution time:     0.489472 ms  
CPU execution time:          73.3571 ms  

N = 100000
Tile_Width = 100
GPU execution time:         126.883 ms
Tiled GPU execution time:     0.005792 ms
CPU execution time:         (timeout)
```

> 🧠 **Conclusion**: Tiling improves GPU performance drastically with large `N` and moderate tile sizes.

---

### 📚 Additional references:

* [Programming Massively Parallel Processors - Chapter 4](https://github.com/R100001/Programming-Massively-Parallel-Processors/tree/master/Chapters/Ch04%20-%20Memory%20And%20Data%20Locality)
* [Simon GPU programming](https://github.com/SzymonOzog/GPU_Programming/tree/main)

---
