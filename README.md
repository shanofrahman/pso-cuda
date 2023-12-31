# pso-cuda
CUDA-baseline implementation of D-Dimensional PSO. Multiple reduction kernels are tested. 

# Particle Swarm Optimization

This code provides a CUDA implementation of Particle Swarm Optimization. Particle Swarm Optimization is a metaheuristic optimization algorithm that is based on simulating the social behavior of bird flocks. It is used to solve optimization problems where an objective function needs to be minimized or maximized.

## Implementation details

The code defines a struct for swarm that holds the particle positions, velocities, and best local positions. For each particle, the code performs a step that involves the calculation of the velocity and position for each dimension. The kernels are called to calculate the fitness function (f) for each particle's calculated position. The fitness values are stored, and the kernel is called for every iteration to update global best, while the local best positions are updated. The global best fitness value is updated if the global minimum is updated.


![ezgif-4-1a7ddd603b](https://github.com/shanofrahman/pso-cuda/assets/77228017/072e6348-f2c5-48ef-8aef-ee725a495c75)

![image](https://github.com/shanofrahman/pso-cuda/assets/77228017/e4436b18-26f1-45bb-a5a8-5f695c318eea)

## How to build and run the code 

To build and run the code follow these instructions:

1. Clone the repository:

   ```
   git clone https://github.com/shanofrahman/pso-cuda
   ```

2. Navigate to the directory

   ```
   cd Particle-swarm-optimization
   ```

3. Compile the code
   
   ```
   make all
   ```

4. Run the executable

   ```
   ./run.sh
   ```

## Output

The default output is the execution time for one iteration. The output is the average execution time for one iteration. The DEBUG flag is defined to print the global best position and value. 
		
## References
1. https://en.wikipedia.org/wiki/Particle_swarm_optimization
