# Makefile for Edifice's CUDA custom-call handlers
#
# Builds priv/cuda/libedifice_xla.so, the XLA FFI library that EXLA loads
# through `EXLA.load_dylib/1` when EDIFICE_FUSED_CUSTOM_CALL=1 is set. The
# usual entry point is `mix edifice.cuda.build --arch sm_XX`, which passes
# the resolved EXLA installation in XLA_EXTENSION_DIR. No EXLA sources or
# cached libraries are modified.
#
# Only the Mamba selective scan (forward + backward, f32) is built. The
# other kernels Edifice once shipped, and the C NIF bridge that ran them on
# eager tensors, were removed on 2026-10-26 after the kernel tier benchmark
# (docs/kernel_tier_benchmarks.md); their sources are kept on the
# `archive/cuda-nif-kernels` branch.
#
# Usage:
#   make ffi          # Build the FFI library (needs XLA_EXTENSION_DIR)
#   make test-compile # Syntax-check the sources without linking
#   make clean        # Remove build artifacts
#   make info         # Show configuration
#
# Override GPU arch:
#   make GPU_ARCH=sm_80   # Ampere
#   make GPU_ARCH=sm_120  # Blackwell (RTX 5090)

# Paths
EDIFICE_ROOT := $(realpath ../..)
BUILD_DIR    := $(EDIFICE_ROOT)/priv/cuda
OBJ_DIR      := build

# Toolchain
NVCC     := nvcc
# Default: sm_75 = Turing (T400, T600, RTX 2xxx)
GPU_ARCH ?= sm_75

# Only the f32 operations exposed by Edifice.Block's CustomCall implementation.
XLA_EXTENSION_DIR ?= $(EDIFICE_ROOT)/deps/exla/cache/xla_extension
FFI_DIR ?= $(OBJ_DIR)/ffi/$(GPU_ARCH)
FFI_SOURCES := fused_selective_scan.cu fused_selective_scan_backward.cu
FFI_OBJECTS := $(patsubst %.cu,$(FFI_DIR)/%.o,$(FFI_SOURCES))
FFI_TARGET := $(BUILD_DIR)/libedifice_xla.so

NVCCFLAGS := -std=c++17 \
             -arch=$(GPU_ARCH) \
             -Xcompiler -fPIC \
             -O3

ifdef DEBUG
	NVCCFLAGS += -g -G -DDEBUG
endif

# Syntax-check objects (no XLA headers; the standalone C ABI of each file)
CHECK_OBJECTS := $(patsubst %.cu,$(OBJ_DIR)/%.o,$(FFI_SOURCES))

ffi: $(FFI_TARGET)

$(FFI_DIR)/%.o: %.cu precision.cuh Makefile $(wildcard $(XLA_EXTENSION_DIR)/include/xla/ffi/api/*.h)
	@mkdir -p $(FFI_DIR)
	$(NVCC) -c $(NVCCFLAGS) -DEXLA_FFI -DNDEBUG -I$(XLA_EXTENSION_DIR)/include $< -o $@

$(FFI_TARGET): $(FFI_OBJECTS) $(XLA_EXTENSION_DIR)/lib/libxla_extension.so FORCE_FFI_LINK
	@mkdir -p $(BUILD_DIR)
	$(NVCC) -shared -arch=$(GPU_ARCH) -o $@.tmp $(FFI_OBJECTS) \
	  -L$(XLA_EXTENSION_DIR)/lib -lxla_extension
	mv $@.tmp $@

$(OBJ_DIR)/%.o: %.cu precision.cuh
	@mkdir -p $(OBJ_DIR)
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

test-compile: $(CHECK_OBJECTS)
	@echo "All kernels compiled successfully"

clean:
	rm -rf $(OBJ_DIR)
	rm -f $(FFI_TARGET)

info:
	@echo "NVCC:              $(NVCC)"
	@echo "GPU_ARCH:          $(GPU_ARCH)"
	@echo "FFI_SOURCES:       $(FFI_SOURCES)"
	@echo "FFI_TARGET:        $(FFI_TARGET)"
	@echo "XLA_EXTENSION_DIR: $(XLA_EXTENSION_DIR)"
	@echo ""
	@$(NVCC) --version

.PHONY: ffi clean test-compile info FORCE_FFI_LINK
