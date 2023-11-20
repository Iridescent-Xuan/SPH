#ifndef __PARTICLES_H__
#define __PARTICLES_H__

#include <string>

namespace xuan {

class Particles {
public:
    ~Particles();
    // init points from obj file
    void InitFromOBJ(const std::string &filename, float3 velocity = make_float3(0.0f, 0.0f, 0.0f));

    void AddPlaneBoundary(float3 start, float3 end, float d, std::string axis);

    // output points to file
    void OutputToOBJ(const std::string &filename) const;
    void OutputToVTK(const std::string &filename, float *scalar = nullptr) const;

    void InitDensity(float *d_density);

    // getters
    int Num() const { return num_; }
    float3 *DPos() const { return d_pos_; }
    float3 *DVel() const { return d_vel_; }
    float *DDensityInit() const { return d_density_init_; }
    bool *DIsBoundary() const { return d_is_boundary_; }

    // reindex
    void Reorder(int *h_index);

    // update position
    void UpdateX(float dt);

    // for neighbor search
    int3 GetCellResolution(float cell_size);
    float3 GetOffset() { return bound_min_; }

private:
    int num_; // total number of points, model + boundary
    int model_num_;
    bool *h_is_boundary_, *d_is_boundary_;
    float3 *h_pos_, *d_pos_;
    float3 *d_vel_, *h_vel_;
    float *d_density_init_, *h_density_init_;
    float3 bound_min_, bound_max_;
};

} // namespace xuan

#endif // __PARTICLE_H__