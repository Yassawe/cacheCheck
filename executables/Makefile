CUDA_INSTALL_PATH ?= /usr/local/cuda-11.5

NVCC := "$(CUDA_INSTALL_PATH)/bin/nvcc"

INCLUDES = -I"$(CUDA_INSTALL_PATH)/include"

EXTRAS_LIB_PATH := /usr/local/cuda-11.5/extras/CUPTI/lib64

LIB_PATH ?= $(CUDA_INSTALL_PATH)/lib64

LIBS = -L $(EXTRAS_LIB_PATH) $(TARGET_CUDA_PATH) -lcuda -L $(LIB_PATH) -lcupti
   
OBJ = o





hardware_counter: hardware_counter.$(OBJ)
	$(NVCC) -o hardware_counter hardware_counter.$(OBJ) $(LIBS) $(INCLUDES)

hardware_counter.$(OBJ): hardware_counter.cu
	$(NVCC) -c $(INCLUDES) $<

hardware_counter.so:
	$(NVCC) --shared --compiler-options '-fPIC' -o hardware_counter.so hardware_counter.cu $(LIBS) $(INCLUDES)

run: hardware_counter
	./$<

clean:
	rm -f hardware_counter hardware_counter.$(OBJ) hardware_counter.so
