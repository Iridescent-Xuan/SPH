#include "simulator.h"
#include <Utilities/cuda_safe_call.h>
#include <Utilities/util.h>
#include <iostream>
#include <string>
#include <vector>

namespace xuan {

void Simulator::InitFromFile(const std::string &file_name) {
    particles_.InitFromOBJ(file_name, parameters_.velocity);
    num_particles_ = particles_.Num();
    AllocateMemory();
}

void Simulator::AddPlaneBoundary(const std::string &axis, float3 start, float3 end) {
    particles_.AddPlaneBoundary(start, end, parameters_.boundary_sample_rate, axis);
    num_particles_ = particles_.Num();
    AllocateMemory();
}

void Simulator::OutputToVTK(const std::string &filename, const std::string &scalar) const {
    if (!scalar.empty()) {
        if (scalar == "density") {
            float *h_density = new float[particles_.Num()];
            CUDA_SAFE_CALL(cudaMemcpy(h_density, d_density_, particles_.Num() * sizeof(float), cudaMemcpyDeviceToHost));
            particles_.OutputToVTK(filename, h_density);
            delete[] h_density;
        } else if (scalar == "velocity") {
            float3 *h_vel = new float3[particles_.Num()];
            CUDA_SAFE_CALL(cudaMemcpy(h_vel, particles_.DVel(), particles_.Num() * sizeof(float3), cudaMemcpyDeviceToHost));
            float *h_vel_magnitude = new float[particles_.Num()];
            for (int i = 0; i < num_particles_; ++i) {
                h_vel_magnitude[i] = sqrtf(h_vel[i].x * h_vel[i].x + h_vel[i].y * h_vel[i].y + h_vel[i].z * h_vel[i].z);
            }
            particles_.OutputToVTK(filename, h_vel_magnitude);
            delete[] h_vel;
            delete[] h_vel_magnitude;
        } else if (scalar == "pressure") {
            float *h_pressure = new float[particles_.Num()];
            CUDA_SAFE_CALL(cudaMemcpy(h_pressure, d_pressure_, particles_.Num() * sizeof(float), cudaMemcpyDeviceToHost));
            particles_.OutputToVTK(filename, h_pressure);
            delete[] h_pressure;
        } else if (scalar == "pressure_force") {
            float3 *h_pressure_force = new float3[particles_.Num()];
            CUDA_SAFE_CALL(cudaMemcpy(h_pressure_force, d_pressure_force_, particles_.Num() * sizeof(float3), cudaMemcpyDeviceToHost));
            float *h_pressure_force_magnitude = new float[particles_.Num()];
            for (int i = 0; i < num_particles_; ++i) {
                h_pressure_force_magnitude[i] = sqrtf(h_pressure_force[i].x * h_pressure_force[i].x + h_pressure_force[i].y * h_pressure_force[i].y + h_pressure_force[i].z * h_pressure_force[i].z);
            }
            particles_.OutputToVTK(filename, h_pressure_force_magnitude);
            delete[] h_pressure_force;
            delete[] h_pressure_force_magnitude;
        } else {
            std::cerr << "Simulator::OutputToVTK: Unknown scalar field: " << scalar << std::endl;
            particles_.OutputToVTK(filename);
        }
    }
}

void Simulator::AllocateMemory() {
    CUDA_SAFE_CALL(cudaFree(d_new_indices_));
    CUDA_SAFE_CALL(cudaMalloc(&d_new_indices_, num_particles_ * sizeof(int)));
    free(h_new_indices_);
    h_new_indices_ = new int[num_particles_];

    d_cell_start_ = d_cell_end_ = nullptr; // varying size

    CUDA_SAFE_CALL(cudaFree(d_density_));
    CUDA_SAFE_CALL(cudaMalloc(&d_density_, num_particles_ * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(d_density_, 0, num_particles_ * sizeof(float)));

    CUDA_SAFE_CALL(cudaFree(d_vel_star_));
    CUDA_SAFE_CALL(cudaMalloc(&d_vel_star_, num_particles_ * sizeof(float3)));

    CUDA_SAFE_CALL(cudaFree(d_pressure_));
    CUDA_SAFE_CALL(cudaMalloc(&d_pressure_, num_particles_ * sizeof(float)));
    CUDA_SAFE_CALL(cudaFree(d_pressure_force_));
    CUDA_SAFE_CALL(cudaMalloc(&d_pressure_force_, num_particles_ * sizeof(float3)));
}

void Simulator::FreeMemory() {
    CUDA_SAFE_CALL(cudaFree(d_kernel_));
    CUDA_SAFE_CALL(cudaFree(d_new_indices_));
    delete[] h_new_indices_;
    CUDA_SAFE_CALL(cudaFree(d_cell_start_));
    CUDA_SAFE_CALL(cudaFree(d_cell_end_));
    CUDA_SAFE_CALL(cudaFree(d_density_));
    CUDA_SAFE_CALL(cudaFree(d_vel_star_));
    CUDA_SAFE_CALL(cudaFree(d_pressure_));
    CUDA_SAFE_CALL(cudaFree(d_pressure_force_));
}

__global__ void knDensity9Cuboid(float *d_density, float3 *d_pos, float3 offset, bool *is_boundary, int *d_cell_start, int *d_cell_end, int num_particles, int3 cell_dim, float cell_size, float mass, CubicSplineKernel *d_W, bool initial, float *d_density_init) {
    int idx_thread = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_particle = idx_thread / 9;
    if (idx_particle >= num_particles)
        return;
    if (!initial && is_boundary[idx_particle])
        return;
    int idx_neighbor = idx_thread % 9;

    float3 pos = d_pos[idx_particle];
    int3 cell = make_int3(floorf((pos.x - offset.x) / cell_size), floorf((pos.y - offset.y) / cell_size), floorf((pos.z - offset.z) / cell_size));
    // 3 cubic at a time
    int neighbor_x_start = max(cell.x - 1, 0);
    int neighbor_x_end = min(cell.x + 1, cell_dim.x - 1);
    int neighbor_y = min(max(cell.y + (idx_neighbor / 3) - 1, 0), cell_dim.y - 1);
    int neighbor_z = min(max(cell.z + (idx_neighbor % 3) - 1, 0), cell_dim.z - 1);
    int neighbor_cell_start = neighbor_x_start + neighbor_y * cell_dim.x + neighbor_z * cell_dim.x * cell_dim.y;
    int neighbor_cell_end = neighbor_x_end + neighbor_y * cell_dim.x + neighbor_z * cell_dim.x * cell_dim.y;
    int neighbor_particle_start = d_cell_start[neighbor_cell_start];
    int neighbor_particle_end = d_cell_end[neighbor_cell_end];

    if (initial) {
        for (int i = neighbor_particle_start; i < neighbor_particle_end; ++i) {
            // in the initial state, only homogeneous particles are considered
            if ((is_boundary[idx_particle] && is_boundary[i]) || (!is_boundary[idx_particle] && !is_boundary[i])) {
                float3 neighbor_pos = d_pos[i];
                float3 r = make_float3(pos.x - neighbor_pos.x, pos.y - neighbor_pos.y, pos.z - neighbor_pos.z);
                float w = d_W->Compute(r);
                atomicAdd(&d_density[idx_particle], mass * w);
            }
        }
    } else {
        for (int i = neighbor_particle_start; i < neighbor_particle_end; ++i) {
            float3 neighbor_pos = d_pos[i];
            float3 r = make_float3(pos.x - neighbor_pos.x, pos.y - neighbor_pos.y, pos.z - neighbor_pos.z);
            float w = d_W->Compute(r);
            // correction term for boundary particles
            if (is_boundary[i] && d_density_init[i] > 1e-6) {
                w *= d_density_init[idx_particle] / d_density_init[i];
            }
            atomicAdd(&d_density[idx_particle], mass * w);
        }
    }
}

void Simulator::Density(bool initial) {
    int3 cell_dim = particles_.GetCellResolution(parameters_.cell_size);

    int block_size = 256;
    int grid_size = GRID_SIZE(num_particles_ * 9, block_size);
    CUDA_SAFE_CALL(cudaMemset(d_density_, 0, num_particles_ * sizeof(float)));

    knDensity9Cuboid<<<grid_size, block_size>>>(d_density_, particles_.DPos(), particles_.GetOffset(), particles_.DIsBoundary(), d_cell_start_, d_cell_end_, num_particles_, cell_dim, parameters_.cell_size, parameters_.mass, d_kernel_, initial, particles_.DDensityInit());

    if (initial)
        particles_.InitDensity(d_density_);
}

__global__ void knAdvection(float3 *v, float3 *v_star, float3 *F, float dt, int num_particles) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_particles)
        return;
    float3 v_star_i = make_float3(v[idx].x + dt * F[idx].x, v[idx].y + dt * F[idx].y, v[idx].z + dt * F[idx].z);
    v_star[idx] = v_star_i;
}

void Simulator::Advection(float3 *other_forces) {
    if (other_forces != nullptr) {
        std::cerr << "Simulator::Advection: other forces are not supported yet." << std::endl;
    }

    float3 *d_F;
    CUDA_SAFE_CALL(cudaMalloc(&d_F, num_particles_ * sizeof(float3)));
    CUDA_SAFE_CALL(cudaMemset(d_F, 0, num_particles_ * sizeof(float3)));

    // gravity
    std::vector<float3> gravity(num_particles_, parameters_.gravity);
    CUDA_SAFE_CALL(cudaMemcpy(d_F, gravity.data(), num_particles_ * sizeof(float3), cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = GRID_SIZE(num_particles_, block_size);
    knAdvection<<<grid_size, block_size>>>(particles_.DVel(), d_vel_star_, d_F, parameters_.dt, num_particles_);
    CUDA_SAFE_CALL(cudaFree(d_F));
}

// p = k * (rho / rho0 - 1)
__global__ void knPressureSE(float *pressure, float *density, float *density0, float k, int num_particles) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_particles)
        return;

    if (density0[idx] > 1e-6)
        pressure[idx] = k * max(density[idx] / density0[idx] - 1.0, 0.0);
}

__global__ void knPressureForce(float3 *force, float *pressure, float *density, float *density0, float3 *d_pos, float3 offset, bool *d_is_boundary, int *d_cell_start, int *d_cell_end, int num_particles, int3 cell_dim, float cell_size, CubicSplineKernel *d_W, float mass) {
    int idx_thread = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_particle = idx_thread / 9;
    if (idx_particle >= num_particles)
        return;
    if (d_is_boundary[idx_particle])
        return;
    int idx_neighbor = idx_thread % 9;

    float3 pos = d_pos[idx_particle];
    int3 cell = make_int3(floorf((pos.x - offset.x) / cell_size), floorf((pos.y - offset.y) / cell_size), floorf((pos.z - offset.z) / cell_size));
    // 3 cubic at a time
    int neighbor_x_start = max(cell.x - 1, 0);
    int neighbor_x_end = min(cell.x + 1, cell_dim.x - 1);
    int neighbor_y = min(max(cell.y + (idx_neighbor / 3) - 1, 0), cell_dim.y - 1);
    int neighbor_z = min(max(cell.z + (idx_neighbor % 3) - 1, 0), cell_dim.z - 1);
    int neighbor_cell_start = neighbor_x_start + neighbor_y * cell_dim.x + neighbor_z * cell_dim.x * cell_dim.y;
    int neighbor_cell_end = neighbor_x_end + neighbor_y * cell_dim.x + neighbor_z * cell_dim.x * cell_dim.y;
    int neighbor_particle_start = d_cell_start[neighbor_cell_start];
    int neighbor_particle_end = d_cell_end[neighbor_cell_end];

    for (int i = neighbor_particle_start; i < neighbor_particle_end; ++i) {
        float3 neighbor_pos = d_pos[i];
        float3 r = make_float3(pos.x - neighbor_pos.x, pos.y - neighbor_pos.y, pos.z - neighbor_pos.z);
        float3 nabla_w = d_W->Gradient(r);
        float coef;
        if (d_is_boundary[i]) {
            if (density0[i] < 1e-6 || density[idx_particle] < 1e-6)
                continue;
            coef = -2.0f * mass * pressure[idx_particle] / powf(density[idx_particle], 2);
            coef *= density0[idx_particle] / density0[i];
        } else {
            if (density[idx_particle] < 1e-6 || density[i] < 1e-6)
                continue;
            coef = -1.0f * mass * (pressure[idx_particle] / powf(density[idx_particle], 2) + pressure[i] / powf(density[i], 2));
        }
        atomicAdd(&force[idx_particle].x, coef * nabla_w.x);
        atomicAdd(&force[idx_particle].y, coef * nabla_w.y);
        atomicAdd(&force[idx_particle].z, coef * nabla_w.z);
    }
}

__global__ void knVelocityCorrection(float3 *d_v, float3 *d_v_star, float3 *d_pressure_force, float dt, int num_particles) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_particles)
        return;
    float3 v_i = make_float3(d_v_star[idx].x + dt * d_pressure_force[idx].x, d_v_star[idx].y + dt * d_pressure_force[idx].y, d_v_star[idx].z + dt * d_pressure_force[idx].z);
    d_v[idx] = v_i;
}

void Simulator::PressureProjection() {
    int block_size = 256;
    int grid_size = GRID_SIZE(num_particles_, block_size);
    // SESPH
    knPressureSE<<<grid_size, block_size>>>(d_pressure_, d_density_, particles_.DDensityInit(), parameters_.k_SE, num_particles_);

    int3 cell_dim = particles_.GetCellResolution(parameters_.cell_size);
    CUDA_SAFE_CALL(cudaMemset(d_pressure_force_, 0, num_particles_ * sizeof(float3)));
    grid_size = GRID_SIZE(num_particles_ * 9, block_size);
    knPressureForce<<<grid_size, block_size>>>(d_pressure_force_, d_pressure_, d_density_, particles_.DDensityInit(), particles_.DPos(), particles_.GetOffset(), particles_.DIsBoundary(), d_cell_start_, d_cell_end_, num_particles_, cell_dim, parameters_.cell_size, d_kernel_, parameters_.mass);

    grid_size = GRID_SIZE(num_particles_, block_size);
    knVelocityCorrection<<<grid_size, block_size>>>(particles_.DVel(), d_vel_star_, d_pressure_force_, parameters_.dt, num_particles_);
}

void Simulator::Run() {
    static bool init_density = true;

    int3 cell_dim = particles_.GetCellResolution(parameters_.cell_size);
    int cell_num = cell_dim.x * cell_dim.y * cell_dim.z;
    CUDA_SAFE_CALL(cudaFree(d_cell_start_));
    CUDA_SAFE_CALL(cudaFree(d_cell_end_));
    CUDA_SAFE_CALL(cudaMalloc(&d_cell_start_, cell_num * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_cell_end_, cell_num * sizeof(int)));

    // reorder particles
    neighbor_list_.CountingSort(d_new_indices_, d_cell_start_, d_cell_end_, particles_.DPos(), particles_.GetOffset(), num_particles_, cell_dim, parameters_.cell_size);
    CUDA_SAFE_CALL(cudaMemcpy(h_new_indices_, d_new_indices_, num_particles_ * sizeof(int), cudaMemcpyDeviceToHost));
    particles_.Reorder(h_new_indices_);

    // compute density
    Density(init_density);
    if (init_density) {
        init_density = false;
    }

    // advection velocity
    Advection();

    // pressure projection
    PressureProjection();

    // update position
    particles_.UpdateX(parameters_.dt);
}

} // namespace xuan