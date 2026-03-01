# Makefile for Edifice fused CUDA kernels
#
# Builds a shared library with C-linkage launch wrappers for each kernel.
# These can be loaded via dlopen / Erlang NIF for standalone testing,
# or via EXLA's device buffer pointer interop.
#
# For XLA FFI integration (EXLA fork), copy the .cu files into
# deps/exla/c_src/exla/custom_calls/ and compile with -DEXLA_FFI.
#
# Usage:
#   make              # Build shared library
#   make clean        # Remove build artifacts
#   make test-compile # Compile only (syntax check)
#   make info         # Show configuration
#
# Override GPU arch:
#   make GPU_ARCH=sm_80   # Ampere
#   make GPU_ARCH=sm_86   # Ampere (RTX 3060-3090)

# Paths
EDIFICE_ROOT := $(realpath ../..)
BUILD_DIR    := $(EDIFICE_ROOT)/priv/cuda
OBJ_DIR      := build

# Toolchain
NVCC     := nvcc
# Default: sm_75 = Turing (T400, T600, RTX 2xxx)
GPU_ARCH ?= sm_75

NVCCFLAGS := -std=c++17 \
             -arch=$(GPU_ARCH) \
             -Xcompiler -fPIC \
             -O3

ifdef DEBUG
	NVCCFLAGS += -g -G -DDEBUG
endif

# Kernel source files (exclude test/bench harnesses — they have main())
KERNEL_SOURCES := fused_mingru_scan.cu fused_minlstm_scan.cu \
                  fused_native_rec_scan.cu fused_liquid_scan.cu
KERNEL_OBJECTS := $(patsubst %.cu,$(OBJ_DIR)/%.o,$(KERNEL_SOURCES))

# All sources for syntax checking
ALL_SOURCES := $(wildcard *.cu)
ALL_OBJECTS := $(patsubst %.cu,$(OBJ_DIR)/%.o,$(ALL_SOURCES))

TARGET      := $(BUILD_DIR)/libedifice_cuda_kernels.so
TEST_BIN    := $(OBJ_DIR)/test_kernels
BENCH_BIN   := $(OBJ_DIR)/bench_kernels

# Default target — shared library only
all: $(TARGET)

$(TARGET): $(KERNEL_OBJECTS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) -shared -arch=$(GPU_ARCH) -o $@ $^
	@echo "Built: $@"

$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

# Test binary — links all kernels + test harness
test-bin: $(TARGET)
	$(NVCC) $(NVCCFLAGS) -o $(TEST_BIN) test_kernels.cu $(KERNEL_SOURCES)
	@echo "Built: $(TEST_BIN)"

# Bench binary — links all kernels + bench harness
bench-bin: $(TARGET)
	$(NVCC) $(NVCCFLAGS) -o $(BENCH_BIN) bench_kernels.cu $(KERNEL_SOURCES)
	@echo "Built: $(BENCH_BIN)"

# Syntax check all sources (including test/bench)
test-compile: $(ALL_OBJECTS)
	@echo "All kernels compiled successfully"

clean:
	rm -rf $(OBJ_DIR)
	rm -f $(TARGET)

# Print configuration
info:
	@echo "NVCC:        $(NVCC)"
	@echo "GPU_ARCH:    $(GPU_ARCH)"
	@echo "KERNELS:     $(KERNEL_SOURCES)"
	@echo "TARGET:      $(TARGET)"
	@echo ""
	@$(NVCC) --version

.PHONY: all clean test-compile test-bin bench-bin info
