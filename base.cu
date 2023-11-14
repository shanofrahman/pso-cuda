// Import necessary libraries
#include <chrono>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "minimum-reduction.cu"
#include "min-reduction-reduce-idle.cu"

// Define macro for error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
    if (abort) exit(code);
  }
}

// Define constants
const int nParticles= 1 << nParticles_pow;
const int D = 2; 

// Define struct for 2D data
struct doubD{
    double data[D];
};

// Define struct for swarm that holds particle positions, velocities and best local positions
struct swarm {
    doubD position[nParticles];
    doubD velocity[nParticles];
    doubD localBest[nParticles];
};

// Define fitness function
__device__ double f(double *position){
    return -10 * (position[0]/5 - pow(position[0], 3) - pow(position[1], 5) ) * exp(-1 * pow(position[0], 2) - pow(position[1], 2) );
}

// Define kernel for one step of particle movement
__global__ void swarmStep(swarm* sw, double *fitnesses, int N, double bGlob, double bLoc, doubD globalBest, int D){
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    
    const double a = 0.72984;
    const int c = 1;
    const int d = 1;
    
    // Initialize random state
    curandState state;
    curand_init(42, p, 0, &state);
    
    // Calculate velocity and position for particle p
    if (p < N){
        double rGlob = ((double)curand_uniform(&state));
        double rLoc = ((double)curand_uniform(&state));
  
        for (int j = 0; j < D; ++j){
            sw->velocity[p].data[j] = a*sw->velocity[p].data[j] + bGlob*rGlob*(globalBest.data[0] - sw->position[p].data[j]) + bLoc*rLoc*(sw->localBest[p].data[j] - sw->position[p].data[j]);
        }

        for (int j = 0; j < D; ++j){
            sw->position[p].data[j] = c*sw->position[p].data[j] + d*sw->velocity[p].data[j];
        }

        // Calculate fitness value for new position and update local best position if better
        double fitness_val = f(sw->position[p].data); // Calculate f
                
        if (f(sw->localBest[p].data) > fitness_val) {
            for (int j = 0; j < D; ++j){
                sw->localBest[p].data[j] = sw->position[p].data[j];
            }
        }
        fitnesses[p] = fitness_val;
    }

}

int main(int argc, char *argv[]) {
    // Define constants for the program
    const int NITER = 10000;
    const double bGlob = 1.49617;
    const double bLoc = 1.49617;

    // Define global best position and value
    doubD globalBest;
    for(int i = 0; i < D; i++){
        globalBest.data[i] = 0.0;
    }
    double globalBestVal = 1000.0;

    // Initialize data structures on device
    swarm *sw;
    double *fitnesses;
    double *fitnesses_reduced;
    int *min_indices;
    gpuErrchk(cudaMallocManaged(&sw, sizeof(swarm)));
    gpuErrchk(cudaMallocManaged(&fitnesses, nParticles*sizeof(double)));
    gpuErrchk(cudaMallocManaged(&fitnesses_reduced, nParticles*sizeof(double)));
    gpuErrchk(cudaMallocManaged(&min_indices, nParticles*sizeof(int)));

    // Initialize particle positions, velocities and local best positions
    for (int i = 0; i < nParticles; ++i) {
        for(int j = 0; j< D; ++j){
            sw->position[i].data[j] = 1.0 * (rand() % int(100))-(100/2);
            sw->velocity[i].data[j] = 1.0;
            sw->localBest[i].data[j] =  1.0 * (rand() % int(100))-(100/2);;
        }
    }
    for(int i=0;i<nParticles;++i) min_indices[i] = i;

    int threads_per_block = 256;
    int blocks_per_grid = nParticles / threads_per_block;
    
    auto start = std::chrono::steady_clock::now();

    // Run swarm optimization for specified number of iterations
    for (int iter = 0; iter < NITER; ++iter) {
        // Move particles one step
        swarmStep<<<blocks_per_grid, threads_per_block>>>(sw, fitnesses, nParticles, bGlob, bLoc, globalBest, D);

        // Use reduction kernel to find minimum fitness value and update global best position and value if necessary
        #ifdef NAIVE_KERNEL
        minReduction<<<blocks_per_grid, threads_per_block>>>(fitnesses, fitnesses_reduced, min_indices);
        minReduction<<<1, threads_per_block>>> (fitnesses_reduced, fitnesses_reduced, min_indices);
        #endif

        #ifdef REDUCE_IDLE_KERNEL
        minReductionReduceIdle<<<blocks_per_grid, threads_per_block>>>(fitnesses, fitnesses_reduced, min_indices);
        minReductionReduceIdle<<<1, threads_per_block>>> (fitnesses_reduced, fitnesses_reduced, min_indices);
        #endif

        cudaDeviceSynchronize();
        
        if (fitnesses_reduced[0] < globalBestVal) {
            globalBestVal = fitnesses_reduced[0];
            globalBest = sw->position[min_indices[0]];
        }
    }
    auto end = std::chrono::steady_clock::now();

    // Print out average execution time for one iteration
    std::cout <<nParticles<<"\t"<<(end-start).count()/NITER/1000.0 << std::endl;

    // Print out global best position and value if DEBUG flag is defined
    #ifdef DEBUG
    std::cout << "The minimum value is --> " << globalBestVal << " and the coordinate is " << globalBest.data[0] << " " << globalBest.data[1] << std::endl;
    #endif

    // Free memory on device
    cudaFree(sw);
    cudaFree(fitnesses);

    return 0;
}