BUILD_DIR = build

targets = $(BUILD_DIR)/base $(BUILD_DIR)/base_cu $(BUILD_DIR)/base_kernel2 


.PHONY: all
all: mk-target-dir $(targets)



mk-target-dir:
	mkdir -p $(BUILD_DIR)


# build rules
$(BUILD_DIR)/base_cu: src/base.cu
	nvcc -O3 -o  $(BUILD_DIR)/base_cu $(CFLAGS) src/base.cu $(EXPERIMENT) -DNAIVE_KERNEL

$(BUILD_DIR)/base_kernel2: src/base.cu
	nvcc -O3 -o  $(BUILD_DIR)/base_kernel2 $(CFLAGS) src/base.cu $(EXPERIMENT) -DREDUCE_IDLE_KERNEL

$(BUILD_DIR)/base: src/base-D.cpp
	nvcc -O3 -o $(BUILD_DIR)/base src/base-D.cpp $(EXPERIMENT)

.PHONY: clean
clean:
	rm $(targets)
