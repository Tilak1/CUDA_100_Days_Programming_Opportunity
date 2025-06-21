# Makefile for CUDA Streams Demo

# Compiler
NVCC = nvcc

# Compiler flags (optimized for A100)
NVCCFLAGS = -std=c++11 -O3 -arch=sm_80 --ptxas-options=-v -use_fast_math

# Target executables
TARGET = cuda_streams_demo
A100_TARGET = a100_optimized_streams

# Source files
SOURCES = cuda_streams_demo.cu
A100_SOURCES = a100_optimized_streams.cu

# Default target
all: $(TARGET) $(A100_TARGET)

# Build targets
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SOURCES)

$(A100_TARGET): $(A100_SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $(A100_TARGET) $(A100_SOURCES)

# Run the demos
run: $(TARGET)
	./$(TARGET)

run-a100: $(A100_TARGET)
	./$(A100_TARGET)

# Clean build files
clean:
    find . -type f -perm -111 -exec rm -f {} +


# Install dependencies (Ubuntu/Debian)
install-deps:
	@echo "Installing CUDA toolkit..."
	@echo "Please install CUDA toolkit manually from NVIDIA website"
	@echo "Or use: sudo apt-get install nvidia-cuda-toolkit"

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version || echo "CUDA not found. Please install CUDA toolkit."
	@nvidia-smi || echo "NVIDIA driver not found. Please install NVIDIA drivers."

.PHONY: all run run-a100 clean install-deps check-cuda