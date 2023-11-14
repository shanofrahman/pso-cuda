
__global__ void minReductionReduceIdle(double *v, double *v_r, int *v_i) {
	// Allocate shared memory
	__shared__ double partial_res[SHMEM_SIZE], indices[SHMEM_SIZE];

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements

    partial_res[threadIdx.x] = MIN(v[i], v[i + blockDim.x]);
	indices[threadIdx.x] = (v[i] < v[i + blockDim.x]) ? (v_i[i]) : (v_i[i + blockDim.x]);

	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_res[threadIdx.x] = MIN(partial_res[threadIdx.x], partial_res[threadIdx.x + s]);
			indices[threadIdx.x] = (partial_res[threadIdx.x] < partial_res[threadIdx.x + s]) ? (indices[threadIdx.x]) : (indices[threadIdx.x + s]);
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_res[0];
		v_i[blockIdx.x] = indices[0];
	}
}

