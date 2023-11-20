#ifndef _NEIGHBOR_LIST_H_
#define _NEIGHBOR_LIST_H_

namespace xuan {
class NeighborList {
public:
    /*
     * @brief Sort particles according to their cell indices.
     * After sorting, particles in the same cell are adjacent.
     * So only the start particle and the end (exclusive) particle of each cell need to be recorded.
     * @param[out] d_new_indices New indices of particles after sorting
     * @param[out] d_cell_start Start index of particles in each cell
     * @param[out] d_cell_end End index (exclusive) of particles in each cell
     * @param[in] d_positions Position of each particle
     * @param[in] offset Offset of the domain
     * @param[in] num_particles Number of particles
     * @param[in] cell_dim Dimension XYZ of the cell grid
     * @param[in] cell_size Size of each cell
     */
    void CountingSort(int *d_new_indices, int *d_cell_start, int *d_cell_end,
                      float3 *d_positions, float3 offset, int num_particles, int3 cell_dim, float cell_size);

    /*
     * @brief Hash particles. Here, we use the cell index as the hash value.
     * @param[out] d_hash Hash value of each particle
     * @param[in] d_positions Position of each particle
     * @param[in] offset Offset of the domain
     * @param[in] num_particles Number of particles
     * @param[in] cell_dim Dimension XYZ of the cell grid
     * @param[in] cell_size Size of each cell
     */
    void Hash(int *d_hash, float3 *d_positions, float3 offset, int num_particles, int3 cell_dim, float cell_size);
};
} // namespace xuan

#endif // _NEIGHBOR_LIST_H_