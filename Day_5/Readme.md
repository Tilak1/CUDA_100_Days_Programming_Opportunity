Day 5 Notes: 
---

# Tiled GEMM: 

Threads collab - Barrier Synch - 

Tiling:  Load a tile to shared mem (whose size is equal to the shared mem size) and the mulitple launched threads which also need the same content - 
will utilize the common tile loaded content to compute a different partial product c[i][j] at the same time. 

It also divides the long access sequences of each thread into phases. and uses barrier synchronization to keep the timing of accesses to each section at close intervals. 
This technique controls the amount of on-chip memory required by localizing the accesses both in time and in space.

The size of the shared memory is quite small, and the capacity of the shared memory should not be exceeded when these M and N elements are loaded into the shared memory. 
This condition can be satisfied by dividing the M and N matrices into smaller tiles so that they can fit into the shared memory.

Now acheiving this threads collab using Tiling


__ syncthreads() is doing barrier synch 

--- code 

Row = by.Tile_width+ty (hereinstead of Bock_dimension - now a reduced Tile_width comes into picture)
Col = bx.Tile_width+tx 

Shared mem loading: 
* What could be the index logic here ? 

Mds = A [][]
Nds = B [][]

from our prev only GEMM   Pvalue += M[Row*Width+k]*N[k*Width+Col]; - now to load in shmem in terms of tiles - we need to instead load in terms of Tile_width - not a single K. 

So, For each tile, k = ph*TILE_WIDTH + tx	
Matrix A, you need to load A[Row][ph*TILE_WIDTH + tx] - flattening to 1D = A[Row * Width + ph * TILE_WIDTH + tx]

Load B[k][Col] from global mem	Same idea, but B is accessed row-wise	d_N[(ph * TILE_WIDTH + ty) * Width + Col]

Store into shared memory	Let all threads in block reuse the tile	Mds[ty][tx], Nds[ty][tx]


__ syncthreads() // All 4 threads are expected to sync well to make sure to read any common content 

final P = Mds[][] * Nds[][]

```cpp
#define TILE_WIDTH 2  // or 16 for larger real examples

__global__ void MatMulTiled(float* A, float* B, float* C, int Width) {
    __shared__ float Asub[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bsub[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        Asub[ty][tx] = A[Row * Width + ph * TILE_WIDTH + tx];
        Bsub[ty][tx] = B[(ph * TILE_WIDTH + ty) * Width + Col];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += Asub[ty][k] * Bsub[k][tx];

        __syncthreads();
    }

    C[Row * Width + Col] = Pvalue;
}
````
Peinding: 

Complete final understanding from PMPP, GPT and Video. 
Run both codes and commit 
-----

Additional references:

* [Programming Massively Parallel Processors - Chapter 4](https://github.com/R100001/Programming-Massively-Parallel-Processors/tree/master/Chapters/Ch04%20-%20Memory%20And%20Data%20Locality)
* [Simon GPU programming](https://github.com/SzymonOzog/GPU_Programming/tree/main)
```
