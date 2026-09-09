#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <utility>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

#include <glm/glm.hpp>
#include <glm/geometric.hpp>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128 // og: 1 << 7 (128)

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f
#define gridCellRatio 1.0f // gridCellWidth / neighborhood distance

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3* dev_pos_sort;
glm::vec3* dev_vel1_sort;

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
float neighborhoodDistance;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // Initialize velocity to 0
  cudaMemset(dev_vel1, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel1 failed!");

  cudaMemset(dev_vel2, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  neighborhoodDistance = std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  gridCellWidth = gridCellRatio * neighborhoodDistance;
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  //gridMinimum.x -= halfGridWidth;
  //gridMinimum.y -= halfGridWidth;
  //gridMinimum.z -= halfGridWidth;
  gridMinimum = glm::vec3(-halfGridWidth, -halfGridWidth, -halfGridWidth);

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.

  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  cudaMalloc((void**)&dev_pos_sort, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_sort failed!");

  cudaMalloc((void**)&dev_vel1_sort, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1_sort");


  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
    glm::vec3 thisPos = pos[iSelf];
    int tot_rule1 = 0;
    int tot_rule3 = 0;
    glm::vec3 avg_pos(0.0f);
    glm::vec3 avg_vel(0.0f);
    glm::vec3 rule2(0.0f);

    for (int i = 0; i < N; ++i) {
        // loop over all neighbor boids
        if (i == iSelf) {
            continue;
        }
        auto dist = glm::length(pos[i] - thisPos);
        if (dist < rule1Distance) {
            ++tot_rule1;
            avg_pos += pos[i];
        }
        if (dist < rule2Distance) {
            rule2 -= (pos[i] - thisPos);
        }
        if (dist < rule3Distance) {
            ++tot_rule3;
            avg_vel += vel[i];
        }
    }
    if (tot_rule1) {
        avg_pos /= tot_rule1;
    }
    else {
        avg_pos = thisPos; // doNothing
    }
    if (tot_rule3) {
        avg_vel /= tot_rule3;
    }
    auto rule1_scaled = (avg_pos - thisPos) * rule1Scale;
    auto rule2_scaled = rule2 * rule2Scale;
    auto rule3_scaled = avg_vel * rule3Scale; // slight deviation from traditional boids

  return rule1_scaled + rule2_scaled + rule3_scaled;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    vel2[idx] = vel1[idx] + computeVelocityChange(N, idx, pos, vel1); // add to cur vel (vel1)
    if (glm::length(vel2[idx]) > maxSpeed) {
        // clamp
        vel2[idx] = glm::normalize(vel2[idx]) * maxSpeed; 
        }

  // Compute a new velocity based on pos and vel1    
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }
    auto thisPos = pos[index];
    auto scaled = glm::floor((thisPos - gridMin) * inverseCellWidth);
    auto gridIdx = gridIndex3Dto1D((int)scaled.x, (int)scaled.y, (int)scaled.z, gridResolution);
    indices[index] = index;
    gridIndices[index] = gridIdx;
    return;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
    //kernResetIntBuffer(N, gridCellStartIndices, -1); // initialize start w -1 for no boids
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) {
        return;
    }
    int gridIdx = particleGridIndices[index];
    if (index == 0) {
        gridCellStartIndices[gridIdx] = index;
        if (N == 1) {
            gridCellEndIndices[gridIdx] = index;
        }
        return;
    }
    int prev_gridIdx = particleGridIndices[index - 1];
    if (prev_gridIdx != gridIdx) {
        gridCellStartIndices[gridIdx] = index;
        gridCellEndIndices[prev_gridIdx] = index - 1;

    }
    if (index == N - 1) {
        gridCellEndIndices[gridIdx] = index;
    }
    return;
}

__global__ void kernSort(int N, int* particleArrayIndices, glm::vec3* pos_in, glm::vec3* vel1_in,
    glm::vec3* pos_out, glm::vec3* vel1_out) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) {
        return;
    }
    auto sort_idx = particleArrayIndices[index];
    pos_out[index] = pos_in[sort_idx];
    vel1_out[index] = vel1_in[sort_idx];
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth, float neighborhoodDist,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) {
        return;
    }

    auto adj_pos = pos[index] - gridMin;

    // consider device helper for grid position
    auto max_grid = glm::floor((adj_pos + glm::vec3(neighborhoodDist)) * inverseCellWidth);
    auto min_grid = glm::floor((adj_pos - glm::vec3(neighborhoodDist)) * inverseCellWidth);
    int z_start = imin(gridResolution - 1, imax(0, (int)min_grid.z));
    int z_end = imin(gridResolution - 1, imax(0, (int)max_grid.z));
    int y_start = imin(gridResolution - 1, imax(0, (int)min_grid.y));
    int y_end = imin(gridResolution - 1, imax(0, (int)max_grid.y));
    int x_start = imin(gridResolution - 1, imax(0, (int)min_grid.x));
    int x_end = imin(gridResolution - 1, imax(0, (int)max_grid.x));

    // our cell grid index
    auto scaled = glm::floor((pos[index] - gridMin) * inverseCellWidth);
    auto gridIdx = gridIndex3Dto1D((int)scaled.x, (int)scaled.y, (int)scaled.z, gridResolution);

    // start the rule1/2/3 accumulations
    glm::vec3 thisPos = pos[index];
    int tot_rule1 = 0;
    int tot_rule3 = 0;
    glm::vec3 avg_pos(0.0f);
    glm::vec3 avg_vel(0.0f);
    glm::vec3 rule2(0.0f);

    // z then y then x (xs are adjacent and should be in order on inner loop
    int g_res_sq = gridResolution * gridResolution;
    int n_gidx; // neighbor grid index
    for (int z = z_start; z <= z_end; ++z) {
        for (int y = y_start; y <= y_end; ++y) {
            n_gidx = z * g_res_sq + y * gridResolution + x_start;
            int end_of_loop = n_gidx + (x_end - x_start);
            for (; n_gidx <= end_of_loop; ++n_gidx) {

                int iter_idx = gridCellStartIndices[n_gidx];
                if (iter_idx < 0) {
                    // no boids inside
                    continue;
                }
                while (iter_idx <= gridCellEndIndices[n_gidx]) {
                    // update velocity here
                    int n_idx = particleArrayIndices[iter_idx];
                    if (index == n_idx) {
                        // this is current boid
                        ++iter_idx;
                        continue;
                    }
                    auto n_pos = pos[n_idx];
                    auto dist = glm::length(n_pos - thisPos);
                    if (dist < rule1Distance) {
                        ++tot_rule1;
                        avg_pos += n_pos;
                    }
                    if (dist < rule2Distance) {
                        rule2 -= (n_pos - thisPos);
                    }
                    if (dist < rule3Distance) {
                        ++tot_rule3;
                        avg_vel += vel1[n_idx];
                    }


                    ++iter_idx;
                }
            }
        }
    }


    if (tot_rule1) {
        avg_pos /= tot_rule1;
    }
    else {
        avg_pos = thisPos;
    }
    if (tot_rule3) {
        avg_vel /= tot_rule3;
    }
    auto rule1_scaled = (avg_pos - thisPos) * rule1Scale;
    auto rule2_scaled = rule2 * rule2Scale;
    auto rule3_scaled = avg_vel * rule3Scale;

    auto vel_change = rule1_scaled + rule2_scaled + rule3_scaled;

    // update vel
    vel2[index] = vel1[index] + vel_change;

    // clamp
    if (glm::length(vel2[index]) > maxSpeed) {
        vel2[index] = glm::normalize(vel2[index]) * maxSpeed;
    }
    return;



}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth, float neighborhoodDist,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {

    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) {
        return;
    }

    auto adj_pos = pos[index] - gridMin;

    // consider device helper for grid position
    auto max_grid = glm::floor((adj_pos + glm::vec3(neighborhoodDist)) * inverseCellWidth);
    auto min_grid = glm::floor((adj_pos - glm::vec3(neighborhoodDist)) * inverseCellWidth);
    int z_start = imin(gridResolution - 1, imax(0, (int)min_grid.z));
    int z_end = imin(gridResolution - 1, imax(0, (int)max_grid.z));
    int y_start = imin(gridResolution - 1, imax(0, (int)min_grid.y));
    int y_end = imin(gridResolution - 1, imax(0, (int)max_grid.y));
    int x_start = imin(gridResolution - 1, imax(0, (int)min_grid.x));
    int x_end = imin(gridResolution - 1, imax(0, (int)max_grid.x));

    // our cell grid index
    auto scaled = glm::floor((pos[index] - gridMin) * inverseCellWidth);
    auto gridIdx = gridIndex3Dto1D((int)scaled.x, (int)scaled.y, (int)scaled.z, gridResolution);

    // start the rule1/2/3 accumulations
    glm::vec3 thisPos = pos[index];
    int tot_rule1 = 0;
    int tot_rule3 = 0;
    glm::vec3 avg_pos(0.0f);
    glm::vec3 avg_vel(0.0f);
    glm::vec3 rule2(0.0f);

    // z then y then x (xs are adjacent and should be in order on inner loop
    int g_res_sq = gridResolution * gridResolution;
    int n_gidx; // neighbor grid index
    for (int z = z_start; z <= z_end; ++z) {
        for (int y = y_start; y <= y_end; ++y) {
            n_gidx = z * g_res_sq + y * gridResolution + x_start;
            int end_of_loop = n_gidx + (x_end - x_start);
            for (; n_gidx <= end_of_loop; ++n_gidx) {

                int iter_idx = gridCellStartIndices[n_gidx];
                if (iter_idx < 0) {
                    // no boids inside
                    continue;
                }
                while (iter_idx <= gridCellEndIndices[n_gidx]) {
                    // update velocity here
                    if (index == iter_idx) {
                        // this is current boid
                        ++iter_idx;
                        continue;
                    }
                    auto n_pos = pos[iter_idx];
                    auto dist = glm::length(n_pos - thisPos);
                    if (dist < rule1Distance) {
                        ++tot_rule1;
                        avg_pos += n_pos;
                    }
                    if (dist < rule2Distance) {
                        rule2 -= (n_pos - thisPos);
                    }
                    if (dist < rule3Distance) {
                        ++tot_rule3;
                        avg_vel += vel1[iter_idx];
                    }


                    ++iter_idx;
                }
            }
        }
    }


    if (tot_rule1) {
        avg_pos /= tot_rule1;
    }
    else {
        avg_pos = thisPos;
    }
    if (tot_rule3) {
        avg_vel /= tot_rule3;
    }
    auto rule1_scaled = (avg_pos - thisPos) * rule1Scale;
    auto rule2_scaled = rule2 * rule2Scale;
    auto rule3_scaled = avg_vel * rule3Scale;

    auto vel_change = rule1_scaled + rule2_scaled + rule3_scaled;

    // update vel
    vel2[index] = vel1[index] + vel_change;

    // clamp
    if (glm::length(vel2[index]) > maxSpeed) {
        vel2[index] = glm::normalize(vel2[index]) * maxSpeed;
    }
    return;
   
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

    kernUpdateVelocityBruteForce << < fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, dev_vel1, dev_vel2);
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);

    // ping pong vel buffers
    std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationScatteredGrid(float dt) {

    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    // gridSideCount == resolution
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum,
        gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

    // sort indices by grid index using thrust library
    dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
    dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects,
            dev_thrust_particleArrayIndices);

    // populate the start index with -1 as a sentinel to skip for no boids
    dim3 cellBlocks((gridCellCount + blockSize - 1) / blockSize);
    kernResetIntBuffer << <cellBlocks, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);

    kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices, dev_gridCellStartIndices,
        dev_gridCellEndIndices);

    kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum,
        gridInverseCellWidth, gridCellWidth, neighborhoodDistance, dev_gridCellStartIndices, dev_gridCellEndIndices,
        dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);

    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
    // pingpong
    std::swap(dev_vel1, dev_vel2);


}

void Boids::stepSimulationCoherentGrid(float dt) {
  
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 cellBlocks((gridCellCount + blockSize - 1) / blockSize);

    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum,
        gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

    // use thrust to sort array indices by grid index
    dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
    dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects,
        dev_thrust_particleArrayIndices);

    // set start indices to default to -1 indicating no boids
    kernResetIntBuffer << <cellBlocks, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);

    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices,
        dev_gridCellStartIndices, dev_gridCellEndIndices);

    // kernel which uses two additional buffers to copy pos and vel1 in order into
    // dev_pos_sort and dev_vel1_sort
    kernSort << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleArrayIndices,
        dev_pos, dev_vel1, dev_pos_sort, dev_vel1_sort);

    // ping pong the "sort" versions into the real versions by swapping pointers
    std::swap(dev_pos, dev_pos_sort);
    std::swap(dev_vel1, dev_vel1_sort);

    // proceed with the coherent grid version, as dev_pos and dev_vel1 are contiguous in grid cell
    kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> > (numObjects,
        gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, neighborhoodDistance, 
        dev_gridCellStartIndices, dev_gridCellEndIndices, dev_pos, dev_vel1, dev_vel2);

    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);

    // ping pong vel buffers
    std::swap(dev_vel1, dev_vel2);

    return;
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  cudaFree(dev_particleGridIndices);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);

  cudaFree(dev_pos_sort);
  cudaFree(dev_vel1_sort);
}

// go back and write this as a start, end, step type unit test. then can input a bunch
// should be quick

void Boids::specUnitTest(std::string name, glm::vec3* before_pos, glm::vec3* before_vel1,
    glm::vec3* before_vel2, glm::vec3* exp_pos, glm::vec3* exp_vel1,
    glm::vec3* exp_vel2, int num_boids, float dt) {
    // wrapper which runs the test in all 3 modes
    specUnitTest(name + " Naive", before_pos, before_vel1, before_vel2, exp_pos, exp_vel1,
        exp_vel2, num_boids, dt, (int)Mode::NAIVE);
    specUnitTest(name + " Scattered", before_pos, before_vel1, before_vel2, exp_pos, exp_vel1,
        exp_vel2, num_boids, dt, (int)Mode::SCATTERED);
    specUnitTest(name + " Coherent", before_pos, before_vel1, before_vel2, exp_pos, exp_vel1,
        exp_vel2, num_boids, dt, (int)Mode::COHERENT);
}

void Boids::specUnitTest(std::string name, glm::vec3* before_pos, glm::vec3* before_vel1,
    glm::vec3* before_vel2, glm::vec3* exp_pos, glm::vec3* exp_vel1,
    glm::vec3* exp_vel2, int num_boids, float dt, int mode) {

    // implementation here

    initSimulation(num_boids);

    // copy to device

    cudaMemcpy(dev_pos, before_pos, num_boids * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vel1, before_vel1, num_boids * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vel2, before_vel2, num_boids * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    // step in given mode

    switch (mode) {
    case NAIVE:
        stepSimulationNaive(dt);
        break;
    case SCATTERED:
        stepSimulationScatteredGrid(dt);
        break;
    case COHERENT:
        stepSimulationCoherentGrid(dt);
        break;
    }

    // copy to host
    std::vector<glm::vec3> pos_res(num_boids);
    std::vector<glm::vec3> vel1_res(num_boids);
    std::vector<glm::vec3> vel2_res(num_boids);
    cudaMemcpy(pos_res.data(), dev_pos, num_boids * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(vel1_res.data(), dev_vel1, num_boids * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(vel2_res.data(), dev_vel2, num_boids * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    // check for error / assertion
    std::cerr << "Running Test: " << name << std::endl;
    int err_count = 0;
    for (int i = 0; i < num_boids; ++i) {
        glm::vec3 pos_dif = pos_res[i] - exp_pos[i];
        float err1 = glm::length(pos_dif);
        glm::vec3 vel1_dif = vel1_res[i] - exp_vel1[i];
        float err2 = glm::length(vel1_dif);
        glm::vec3 vel2_dif = vel2_res[i] - exp_vel2[i];
        float err3 = glm::length(vel2_dif);
        if (err1 > 0.001 || err2 > 0.001 || err3 > 0.001) {
            std::cerr << name << " FAIL: Boid " << i << " pos: " << err1 << " vel1: " << err2 <<
                " vel2: " << err3 << std::endl;
            ++err_count;
        }
    }
    std::cerr << "Test complete: " << err_count << " errors." << std::endl;

    // clean
    endSimulation();

    return;


}

void Boids::unitTest() {

  // Simple unit test, 3 in flock 1 outside, 0 velocity initialization
    glm::vec3 test1_pos[4] = {
        glm::vec3(0.0f),
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 2.0f, 0.0f),
        glm::vec3(50.0f)
    };
    glm::vec3 test1_vel[4] = {
        glm::vec3(0.0f),
        glm::vec3(0.0f),
        glm::vec3(0.0f),
        glm::vec3(0.0f)
    };
    glm::vec3 test1_vel2[4] = {
        glm::vec3(0.0f),
        glm::vec3(0.0f),
        glm::vec3(0.0f),
        glm::vec3(0.0f)
    };
    glm::vec3 pos_exp[4] = {
        glm::vec3(-0.095f, -0.19f, 0.0f),
        glm::vec3(1.19f, -0.19f, 0.0f),
        glm::vec3(-0.095f, 2.38f, 0.0f),
        glm::vec3(50.0f)
    };
    // dev1 pts to dev2 from the simulation
    glm::vec3 vel1_exp[4] = {
    glm::vec3(-0.095f, -0.19f, 0.0f),
    glm::vec3(0.19f, -0.19f, 0.0f),
    glm::vec3(-0.095f, 0.38f, 0.0f),
    glm::vec3(0.0f)
    };
    // dev2 is "empty" it was the existing dev1
    glm::vec3 vel2_exp[4] = {
        glm::vec3(0.0f),
        glm::vec3(0.0f),
        glm::vec3(0.0f),
        glm::vec3(0.0f)
    };

    // run test
    specUnitTest("Simple test", test1_pos, test1_vel, test1_vel2,
        pos_exp, vel1_exp, vel2_exp, 4, 1.0);

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
