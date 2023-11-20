#include "neighbor_list.h"

#include <Utilities/cuda_safe_call.h>
#include <Utilities/util.h>

#include <thrust/scan.h>

namespace xuan {
__global__ void knHash(int *d_hash, float3 *d_positions, float3 offset, int num_particles, int3 cell_dim, float cell_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles) {
        // TODO: other hash function?
        int3 cell = make_int3(
            floorf((d_positions[i].x - offset.x) / cell_size),
            floorf((d_positions[i].y - offset.y) / cell_size),
            floorf((d_positions[i].z - offset.z) / cell_size));
        d_hash[i] = cell.x + cell.y * cell_dim.x + cell.z * cell_dim.x * cell_dim.y;
    }
}

void NeighborList::Hash(int *d_hash, float3 *d_positions, float3 offset, int num_particles, int3 cell_dim, float cell_size) {
    int block_size = 256;
    int grid_size = GRID_SIZE(num_particles, block_size);
    knHash<<<grid_size, block_size>>>(d_hash, d_positions, offset, num_particles, cell_dim, cell_size);
}

__global__ void knAtomicAdd(int *d_cell_capacity, int *d_particle_offset, int *d_hash, int num_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // the ith particle
    if (i < num_particles) {
        d_particle_offset[i] = atomicAdd(&d_cell_capacity[d_hash[i]], 1);
    }
}

__global__ void knReorder(int *d_new_indices, int *d_cell_accumulate, int *d_particle_offset, int *d_hash, int num_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // the ith particle
    if (i < num_particles) {
        d_new_indices[i] = d_particle_offset[i] + d_cell_accumulate[d_hash[i]];
    }
}

__global__ void knMyMemsetScalar(int *target, int value) {
    *target = value;
}

void NeighborList::CountingSort(int *d_new_indices, int *d_cell_start, int *d_cell_end,
                                float3 *d_positions, float3 offset, int num_particles, int3 cell_dim, float cell_size) {
    int cell_num = cell_dim.x * cell_dim.y * cell_dim.z;
    // 1. hash
    int *d_hash;
    CUDA_SAFE_CALL(cudaMalloc(&d_hash, num_particles * sizeof(int)));
    Hash(d_hash, d_positions, offset, num_particles, cell_dim, cell_size);

    // 2. counting
    // 2.1 atomic add
    int *d_cell_capacity;
    CUDA_SAFE_CALL(cudaMalloc(&d_cell_capacity, cell_num * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_cell_capacity, 0, cell_num * sizeof(int)));
    int *d_particle_offset;
    CUDA_SAFE_CALL(cudaMalloc(&d_particle_offset, num_particles * sizeof(int)));
    int block_size = 256;
    int grid_size = GRID_SIZE(num_particles, block_size);
    knAtomicAdd<<<grid_size, block_size>>>(d_cell_capacity, d_particle_offset, d_hash, num_particles);

    // 2.2 prefix sum
    int *d_cell_accumulate;
    CUDA_SAFE_CALL(cudaMalloc(&d_cell_accumulate, cell_num * sizeof(int)));
    thrust::exclusive_scan(thrust::device, d_cell_capacity, d_cell_capacity + cell_num, d_cell_accumulate);

    // 3. reorder
    grid_size = GRID_SIZE(num_particles, block_size);
    knReorder<<<grid_size, block_size>>>(d_new_indices, d_cell_accumulate, d_particle_offset, d_hash, num_particles);

    // 4. build up linked list
    CUDA_SAFE_CALL(cudaMemcpy(d_cell_start, d_cell_accumulate, cell_num * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_cell_end, d_cell_accumulate + 1, (cell_num - 1) * sizeof(int), cudaMemcpyDeviceToDevice));
    // CUDA_SAFE_CALL(cudaMemset(d_cell_end + cell_num - 1, num_particles, sizeof(int))); // WRONG!
    knMyMemsetScalar<<<1, 1>>>(d_cell_end + cell_num - 1, num_particles);

    // 5. free
    CUDA_SAFE_CALL(cudaFree(d_hash));
    CUDA_SAFE_CALL(cudaFree(d_cell_capacity));
    CUDA_SAFE_CALL(cudaFree(d_particle_offset));
    CUDA_SAFE_CALL(cudaFree(d_cell_accumulate));
}
} // namespace xuan