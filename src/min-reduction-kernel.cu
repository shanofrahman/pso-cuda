#define SHMEM_SIZE 256
#define MIN(a,b) (((a)<(b))?(a):(b))

__global__ void minReduction(double *v, double *v_r, int *v_i) {
	// Allocate shared memory
	__shared__ double partial_res[SHMEM_SIZE], indices[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements into shared memory
	partial_res[threadIdx.x] = v[tid];
	indices[threadIdx.x] = v_i[tid];
	__syncthreads();

	// Iterate of log base 2 the block dimension
	for (int s = 1; s < blockDim.x; s *= 2) {
		// Reduce the threads performing work by half previous the previous
		// iteration each cycle
		if (threadIdx.x % (2 * s) == 0) {
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