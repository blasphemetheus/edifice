# Makefile for Edifice fused CUDA kernels
#
# Builds shared libraries with C-linkage launch wrappers for each kernel.
# Two variants: f32 (default) and bf16 (mixed precision with f32 accumulators).
# These can be loaded via dlopen / Erlang NIF for standalone testing,
# or via EXLA's device buffer pointer interop.
#
# For XLA FFI integration (EXLA fork), copy the .cu files into
# deps/exla/c_src/exla/custom_calls/ and compile with -DEXLA_FFI.
#
# Usage:
#   make              # Build both f32 and bf16 shared libraries
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
                  fused_native_rec_scan.cu fused_liquid_scan.cu \
                  fused_linear_scan.cu fused_delta_rule_scan.cu \
                  fused_delta_product_scan.cu fused_slstm_scan.cu \
                  fused_lstm_scan.cu fused_gru_scan.cu \
                  fused_ttt_scan.cu fused_selective_scan.cu \
                  fused_kda_scan.cu fused_rla_scan.cu \
                  fused_flash_attention.cu \
                  fused_laser_attention.cu fused_fox_attention.cu \
                  fused_reservoir_scan.cu fused_titans_scan.cu \
                  fused_miras_scan.cu fused_gsa_scan.cu \
                  fused_linear_scan_backward.cu \
                  fused_mingru_scan_backward.cu \
                  fused_minlstm_scan_backward.cu \
                  fused_elu_gru_scan_backward.cu \
                  fused_real_gru_scan_backward.cu \
                  fused_diag_linear_scan_backward.cu \
                  fused_lstm_scan_backward.cu \
                  fused_gru_scan_backward.cu \
                  fused_liquid_scan_backward.cu \
                  fused_delta_rule_scan_backward.cu \
                  fused_gated_delta_net_scan_backward.cu \
                  fused_delta_product_scan_backward.cu \
                  fused_slstm_scan_backward.cu \
                  fused_selective_scan_backward.cu \
                  fused_kda_scan_backward.cu \
                  fused_rla_scan_backward.cu \
                  fused_ttt_scan_backward.cu \
                  fused_mingru_block_scan.cu \
                  fused_minlstm_block_scan.cu \
                  fused_flash_attention_backward.cu \
                  fused_laser_attention_backward.cu \
                  fused_fox_attention_backward.cu

# f32 objects (default compilation)
KERNEL_OBJECTS     := $(patsubst %.cu,$(OBJ_DIR)/%.o,$(KERNEL_SOURCES))

# bf16 objects (compiled with -DUSE_BF16)
KERNEL_OBJECTS_BF16 := $(patsubst %.cu,$(OBJ_DIR)/%_bf16.o,$(KERNEL_SOURCES))

# All sources for syntax checking
ALL_SOURCES := $(wildcard *.cu)
ALL_OBJECTS := $(patsubst %.cu,$(OBJ_DIR)/%.o,$(ALL_SOURCES))

TARGET      := $(BUILD_DIR)/libedifice_cuda_kernels.so
TARGET_BF16 := $(BUILD_DIR)/libedifice_cuda_kernels_bf16.so
TEST_BIN    := $(OBJ_DIR)/test_kernels
BENCH_BIN   := $(OBJ_DIR)/bench_kernels

# Default target — both f32 and bf16 shared libraries
all: $(TARGET) $(TARGET_BF16)

# f32 shared library
$(TARGET): $(KERNEL_OBJECTS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) -shared -arch=$(GPU_ARCH) -o $@ $^
	@echo "Built: $@"

# bf16 shared library
$(TARGET_BF16): $(KERNEL_OBJECTS_BF16)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) -shared -arch=$(GPU_ARCH) -o $@ $^
	@echo "Built: $@"

# f32 object rule (depends on precision.cuh)
$(OBJ_DIR)/%.o: %.cu precision.cuh
	@mkdir -p $(OBJ_DIR)
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

# bf16 object rule (depends on precision.cuh, compiled with -DUSE_BF16)
$(OBJ_DIR)/%_bf16.o: %.cu precision.cuh
	@mkdir -p $(OBJ_DIR)
	$(NVCC) -c $(NVCCFLAGS) -DUSE_BF16 $< -o $@

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
	rm -f $(TARGET) $(TARGET_BF16)

# Print configuration
info:
	@echo "NVCC:        $(NVCC)"
	@echo "GPU_ARCH:    $(GPU_ARCH)"
	@echo "KERNELS:     $(KERNEL_SOURCES)"
	@echo "TARGET_F32:  $(TARGET)"
	@echo "TARGET_BF16: $(TARGET_BF16)"
	@echo ""
	@$(NVCC) --version

.PHONY: all clean test-compile test-bin bench-bin info
