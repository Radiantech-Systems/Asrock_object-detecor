CUDA_VER ?= 12.6

CC ?= g++
CXX ?= g++
NVCC ?= nvcc

SYSROOT ?=

# ============================================================
# Yocto compiler handling
# ============================================================

# BitBake passes CXX as:
# aarch64-oe4t-linux-g++ <flags> --sysroot=<recipe-sysroot>
#
# NVCC needs only the compiler executable for -ccbin.
NVCC_CXX := $(word 1,$(CXX))

# Keep the remaining Yocto CXX flags.
# This includes the Yocto target compiler options and sysroot.
NVCC_HOST_FLAGS := $(filter-out $(NVCC_CXX),$(CXX))


# ============================================================
# Target sysroot paths
# ============================================================

DEEPSTREAM_DIR := $(SYSROOT)/opt/nvidia/deepstream/deepstream-7.1
CUDA_DIR       := $(SYSROOT)/usr/local/cuda-$(CUDA_VER)
TENSORRT_DIR   := $(SYSROOT)/usr/include
TARGET_LIB_DIR := $(SYSROOT)/usr/lib


# ============================================================
# C/C++ compiler flags
# ============================================================

CFLAGS := -Wall
CFLAGS += -std=c++11
CFLAGS += -fPIC
CFLAGS += -Wno-error=deprecated-declarations

# DeepStream headers
CFLAGS += -I$(DEEPSTREAM_DIR)/sources/includes

# TensorRT headers
CFLAGS += -I$(TENSORRT_DIR)

# CUDA headers
CFLAGS += -I$(CUDA_DIR)/include


# ============================================================
# CUDA compiler flags
# ============================================================

CUFLAGS := -Xcompiler -fPIC

# DeepStream headers
CUFLAGS += -I$(DEEPSTREAM_DIR)/sources/includes

# TensorRT headers
CUFLAGS += -I$(TENSORRT_DIR)

# CUDA headers
CUFLAGS += -I$(CUDA_DIR)/include


# ============================================================
# Libraries
# ============================================================

LIBS := -L$(DEEPSTREAM_DIR)/lib
LIBS += -L$(TARGET_LIB_DIR)
LIBS += -L$(CUDA_DIR)/lib
LIBS += -L$(CUDA_DIR)/lib64

# TensorRT
LIBS += -lnvinfer_plugin
LIBS += -lnvinfer
LIBS += -lnvonnxparser

# CUDA
LIBS += -lcudart
LIBS += -lcublas

# C++ filesystem
LIBS += -lstdc++fs


# ============================================================
# Shared library linker flags
# ============================================================

LDFLAGS_SO := -shared


# ============================================================
# Source files
# ============================================================

# Exclude calibrator.cpp
SRCFILES := $(filter-out calibrator.cpp,$(wildcard *.cpp))

# Layer implementations
SRCFILES += $(wildcard layers/*.cpp)

# CUDA source files
SRCFILES += $(wildcard *.cu)


# ============================================================
# Output
# ============================================================

TARGET_LIB := libnvdsinfer_custom_impl_Yolo.so


# ============================================================
# Object files
# ============================================================

TARGET_OBJS := $(SRCFILES:.cpp=.o)
TARGET_OBJS := $(TARGET_OBJS:.cu=.o)


# ============================================================
# Targets
# ============================================================

.PHONY: all clean

all: $(TARGET_LIB)


# ============================================================
# Compile C++ files
# ============================================================

%.o: %.cpp Makefile
	$(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<


# ============================================================
# Compile CUDA files
# ============================================================

# Use Yocto AArch64 compiler as NVCC host compiler.
# Yocto compiler configuration supplies the target sysroot.
%.o: %.cu Makefile
	$(NVCC) \
		-ccbin "$(NVCC_CXX)" \
		-Xcompiler "$(NVCC_HOST_FLAGS)" \
		$(CUFLAGS) \
		$(CPPFLAGS) \
		-c -o $@ $<


# ============================================================
# Link YOLO26 custom parser
# ============================================================

$(TARGET_LIB): $(TARGET_OBJS)
	$(CXX) $(LDFLAGS_SO) -o $@ $(TARGET_OBJS) $(LDFLAGS) $(LIBS)


# ============================================================
# Clean
# ============================================================

clean:
	rm -f $(TARGET_LIB)
	rm -f $(TARGET_OBJS)
