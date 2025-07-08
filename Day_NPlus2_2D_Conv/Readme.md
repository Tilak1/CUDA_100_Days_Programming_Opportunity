Theory or equations: 

# CUDA 2D Convolution Implementation

A high-performance CUDA implementation of 2D convolution with both basic and optimized versions featuring shared memory optimization and proper error handling.

## 🏃 Usage

### Basic Version
```bash
./basic_conv
```

### Improved Version
```bash
./improved_conv
```

### Sample Output
```
2D Convolution with 1024x1024 matrix and 7x7 kernel
Initializing data...
Copying data to GPU...
Grid: 64x64, Block: 16x16
Improved kernel execution time: 45 ms
Shared memory kernel execution time: 32 ms
Copying result back to CPU...
Verifying results...
Verification PASSED!
COMPLETED SUCCESSFULLY!
```

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
- ✅ **Memory management**: RAII-style cleanup

### Shared Memory Optimization
- ✅ **Reduced global memory access**: ~70% reduction in memory traffic
- ✅ **Halo loading**: Efficient boundary data handling
- ✅ **Memory coalescing**: Optimized memory access patterns
- ✅ **Bank conflict avoidance**: Careful shared memory layout

## 🐛 Common Issues

### Compilation Errors
```bash
# If you get "undefined reference to cudaXXX"
nvcc -o conv conv.cu -lcuda -lcudart

# For older CUDA versions
nvcc -o conv conv.cu -std=c++11 -arch=sm_35
```

### Runtime Errors
- **"Out of memory"**: Reduce matrix size or use smaller GPU
- **"Invalid device function"**: Check GPU compute capability
- **"Kernel launch failed"**: Verify grid/block dimensions

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

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- CUDA optimization techniques from NVIDIA Developer documentation

- Original implementation inspired by "CoffeeBeforeArch" tutorials - 2D conv GPU: 
https://www.youtube.com/watch?v=qxcfco89wvs


-2D Conv CPU: 
https://pub.towardsai.net/deep-learning-from-scratch-in-modern-c-convolutions-5c55598473e9
