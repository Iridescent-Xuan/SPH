#ifndef __SIMULATION_H__
#define __SIMULATION_H__

#include <Discretization/kernel.cuh>
#include <NeighborList/neighbor_list.h>
#include <Particle/particles.h>
#include <Utilities/parameter.h>

#include <iostream>

namespace xuan {

class Simulator {
public:
    Simulator(SysParameter parameter = SysParameter()) : parameters_(parameter) {
        if (parameters_.kernel == "cubic") {
            kernel_ = new CubicSplineKernel(parameters_.h);
            CUDA_SAFE_CALL(cudaMalloc((void **)&d_kernel_, sizeof(CubicSplineKernel)));
            CUDA_SAFE_CALL(cudaMemcpy(d_kernel_, kernel_, sizeof(CubicSplineKernel), cudaMemcpyHostToDevice));
        } else {
            std::cerr << "Unsupported kernel: " << parameters_.kernel << std::endl;
            std::cerr << "Choose cubic spline kernel instead." << std::endl;
            kernel_ = new CubicSplineKernel(parameters_.h);
            CUDA_SAFE_CALL(cudaMalloc((void **)&d_kernel_, sizeof(CubicSplineKernel)));
            CUDA_SAFE_CALL(cudaMemcpy(d_kernel_, kernel_, sizeof(CubicSplineKernel), cudaMemcpyHostToDevice));
        }
    }

    ~Simulator() {
        delete kernel_;
        FreeMemory();
    }

    void InitFromFile(const std::string &filename);

    void AddPlaneBoundary(const std::string &axis, float3 start = make_float3(0.0f, 0.0f, 0.0f), float3 end = make_float3(0.0f, 0.0f, 0.0f));

    // supported scalar fields: density...
    void OutputToVTK(const std::string &filename, const std::string &scalar = "") const;

    int NumParticles() const { return particles_.Num(); }

    void Run();

private:
    CubicSplineKernel *kernel_ = nullptr;
    NeighborList neighbor_list_;
    Particles particles_;
    int num_particles_;
    SysParameter parameters_;
    // memory management
    CubicSplineKernel *d_kernel_ = nullptr;
    int *d_new_indices_ = nullptr;
    int *h_new_indices_ = nullptr;
    int *d_cell_start_ = nullptr;
    int *d_cell_end_ = nullptr;
    float *d_density_ = nullptr;
    float3 *d_vel_star_ = nullptr;
    float *d_pressure_ = nullptr;
    float3 *d_pressure_force_ = nullptr;
    void AllocateMemory();
    void FreeMemory();

    void Density(bool initial = false);
    void Advection(float3 *other_forces = nullptr);
    void PressureProjection();
};

} // namespace xuan

#endif // __SIMULATION_H__