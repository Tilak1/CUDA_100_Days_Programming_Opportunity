# CUDA 2D Convolution Implementation

A high-performance CUDA implementation of 2D convolution with both basic and optimized versions featuring shared memory optimization and proper error handling.


## 🧠 Theory and Implementation Details

### Grid and Block Configuration
For this implementation with 16×16 block size and 7×7 mask:
```cpp
TILE_SIZE = 16              // Block dimensions (16×16 threads)
MASK_DIM = 7               // Convolution kernel size
MASK_OFFSET = 3            // (MASK_DIM / 2)
Shared Memory Size = 22×22  // (TILE_SIZE + 2*MASK_OFFSET)²
```

### Thread-to-Memory Mapping
```cpp
// Global matrix coordinates
row = blockIdx.y * TILE_SIZE + threadIdx.y
col = blockIdx.x * TILE_SIZE + threadIdx.x

// Shared memory coordinates  
tile_row = threadIdx.y  // 0 to 15
tile_col = threadIdx.x  // 0 to 15
```

### Main Tile Loading Strategy
The main tile establishes the **reference coordinate system** at the center:
```cpp
tile[tile_row + MASK_OFFSET][tile_col + MASK_OFFSET] = matrix[row * N + col];
//   ↑ Center placement (rows 3-18, cols 3-18)
```

### Halo Loading Index Mapping
Each halo region is positioned relative to the main tile center:

![image](https://github.com/user-attachments/assets/520c6a43-0db9-47dd-89cc-917a179cd3cb)


| Halo Position | Shared Memory Index | Global Memory Index |
|---------------|---------------------|---------------------|
| **Top** | `[tile_row][tile_col + MASK_OFFSET]` | `[row - MASK_OFFSET][col]` |
| **Bottom** | `[tile_row + 2*MASK_OFFSET][tile_col + MASK_OFFSET]` | `[row + MASK_OFFSET][col]` |
| **Left** | `[tile_row + MASK_OFFSET][tile_col]` | `[row][col - MASK_OFFSET]` |
| **Right** | `[tile_row + MASK_OFFSET][tile_col + 2*MASK_OFFSET]` | `[row][col + MASK_OFFSET]` |

![image](https://github.com/user-attachments/assets/5a767857-b5df-4434-93a5-9cfcef6246c1)

This above mapping image clarifies how tile indexing is chanaged w.r.t main / global memory indexing. 

### Memory Layout Visualization
```
Shared Memory tile[22][22] Layout:
    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
 0  .  .  .  T  T  T  T  T  T  T  T  T  T  T  T  T  T  T  T  .  .  .
 1  .  .  .  T  T  T  T  T  T  T  T  T  T  T  T  T  T  T  T  .  .  .
 2  .  .  .  T  T  T  T  T  T  T  T  T  T  T  T  T  T  T  T  .  .  .
 3  L  L  L  M  M  M  M  M  M  M  M  M  M  M  M  M  M  M  M  R  R  R
 4  L  L  L  M  M  M  M  M  M  M  M  M  M  M  M  M  M  M  M  R  R  R
 ...
18  L  L  L  M  M  M  M  M  M  M  M  M  M  M  M  M  M  M  M  R  R  R
19  .  .  .  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  .  .  .
20  .  .  .  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  .  .  .
21  .  .  .  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  .  .  .

Legend: M=Main, T=Top, B=Bottom, L=Left, R=Right, .=Unused
```

### Multi-Element Loading Per Thread
Each thread can load multiple elements depending on its position:
- **Corner threads**: Load main tile + 2 halo regions
- **Edge threads**: Load main tile + 1 halo region  
- **Center threads**: Load main tile only

### Generic Halo Loading Formula
```cpp
// Universal indexing pattern for all halos:
shmem[Tx ± offset_row][Ty ± offset_col] = Array[row ± offset_row][col ± offset_col]

// Where offsets are chosen to:
// 1. Place halo in correct shared memory region
// 2. Load from correct global memory region
// 3. Maintain spatial relationships for convolution
```
Future Work
### Future Enhancements
- [ ] **Bank conflict analysis**: Optimize shared memory access patterns
- [ ] **Warp-level optimizations**: Leverage cooperative groups
- [ ] **Dynamic kernel size**: Runtime configurable mask dimensions
- [ ] **Multi-GPU support**: Distribute computation across multiple GPUs

## 🏗️ Architecture

### Basic Implementation (`basic_conv.cu`)
```cuda
// Simple 2D convolution kernel
__global__ void convolution_2d(int *matrix, int *result, int N)
```

### Improved Implementation (`improved_conv.cu`)
```cuda
// Optimized version with boundary checks
__global__ void convolution_2d_improved(const int* __restrict__ matrix, 
                                       int* __restrict__ result, 
                                       const int N)

// Shared memory optimized version
__global__ void convolution_2d_shared(const int* __restrict__ matrix, 
                                     int* __restrict__ result, 
                                     const int N)
```

## ⚙️ Configuration

### Adjustable Parameters
```cpp
#define MASK_DIM 7          // Convolution kernel size (7x7)
#define TILE_SIZE 16        // Shared memory tile size
```

### Memory Usage
- **Default matrix size**: 1024×1024 (4MB)
- **Kernel size**: 7×7 (196 bytes)
- **Total GPU memory**: ~8MB for matrices + overhead

## 📊 Performance Comparison

| Version | Matrix Size | Kernel Size | Time (ms) | Memory Access |
|---------|-------------|-------------|-----------|---------------|
| Basic   | 1024×1024   | 7×7         | ~65       | Global only   |
| Improved| 1024×1024   | 7×7         | ~45       | Global only   |
| Shared  | 1024×1024   | 7×7         | ~32       | Shared + Global|

## 🔍 Key Improvements

### From Basic to Improved
- ✅ **Boundary checking**: Prevents out-of-bounds memory access
- ✅ **Error handling**: Comprehensive CUDA error checking
- ✅ **Synchronization**: Proper device synchronization
- ✅ **Loop unrolling**: Performance optimization hints
- ✅ **Memory management**: RAII-style (std::unique pointer) cleanup

### Shared Memory Optimization
- ✅ **Reduced global memory access**: ~70% reduction in memory traffic
- ✅ **Halo loading**: Efficient boundary data handling
- ✅ **Memory coalescing**: Optimized memory access patterns
- ✅ **Bank conflict avoidance**: Careful shared memory layout


## 🧪 Testing

### Verification Process
1. **CPU Reference**: Computes expected result on CPU
2. **Element-wise comparison**: Validates every output element
3. **Boundary testing**: Verifies edge case handling
4. **Performance measurement**: Reports execution times

### Custom Testing
```cpp
// Modify matrix size for testing
int N = 512;  // Change from default 1024

// Modify kernel size
#define MASK_DIM 5  // Change from default 7
```

## 📈 Optimization Techniques Used

1. **Constant Memory**: Fast read-only kernel storage
2. **Shared Memory**: On-chip memory for frequently accessed data
3. **Memory Coalescing**: Optimized global memory access patterns
4. **Loop Unrolling**: Reduced loop overhead
5. **Restrict Pointers**: Compiler optimization hints
6. **Proper Synchronization**: Avoiding race conditions

## 📚 Further Reading

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Convolution in Computer Vision](https://en.wikipedia.org/wiki/Convolution)
- [Shared Memory Optimization](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)

## 🙏 Acknowledgments

- Original implementation inspired by "CoffeeBeforeArch" tutorials (https://github.com/CoffeeBeforeArch/cuda_programming/tree/master/05_convolution/2d_constant_memory)


