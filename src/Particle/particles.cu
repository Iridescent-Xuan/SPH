#include "particles.h"
#include <Utilities/cuda_safe_call.h>
#include <Utilities/util.h>

#include <fstream>
#include <vector>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include <cassert>

namespace xuan {

Particles::~Particles() {
    delete[] h_pos_;
    delete[] h_vel_;
    CUDA_SAFE_CALL(cudaFree(d_pos_));
    CUDA_SAFE_CALL(cudaFree(d_vel_));
    CUDA_SAFE_CALL(cudaFree(d_density_init_));
}

void Particles::InitFromOBJ(const std::string &filename, float3 velocity) {
    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        exit(1);
    }

    std::vector<float3> vertices;

    float x_min, y_min, z_min;
    x_min = y_min = z_min = std::numeric_limits<float>::max();
    float x_max, y_max, z_max;
    x_max = y_max = z_max = std::numeric_limits<float>::min();

    std::string line;
    while (std::getline(fin, line)) {
        if (line.substr(0, 2) == "v ") {
            std::istringstream s(line.substr(2));
            float3 v;
            s >> v.x;
            s >> v.y;
            s >> v.z;
            vertices.push_back(v);

            x_min = std::min(x_min, v.x);
            y_min = std::min(y_min, v.y);
            z_min = std::min(z_min, v.z);
            x_max = std::max(x_max, v.x);
            y_max = std::max(y_max, v.y);
            z_max = std::max(z_max, v.z);
        }
    }

    bound_min_ = make_float3(x_min, y_min, z_min);
    bound_max_ = make_float3(x_max, y_max, z_max);

    num_ = model_num_ = vertices.size();
    h_pos_ = new float3[num_];
    CUDA_SAFE_CALL(cudaMemcpy(h_pos_, vertices.data(), sizeof(float3) * num_, cudaMemcpyHostToHost));
    CUDA_SAFE_CALL(cudaMalloc(&d_pos_, sizeof(float3) * num_));
    CUDA_SAFE_CALL(cudaMemcpy(d_pos_, h_pos_, sizeof(float3) * num_, cudaMemcpyHostToDevice));
    h_vel_ = new float3[num_];
    for (int i = 0; i < num_; ++i)
        h_vel_[i] = velocity;
    CUDA_SAFE_CALL(cudaMalloc(&d_vel_, sizeof(float3) * num_));
    CUDA_SAFE_CALL(cudaMemcpy(d_vel_, h_vel_, sizeof(float3) * num_, cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc(&d_density_init_, sizeof(float) * num_));
    CUDA_SAFE_CALL(cudaMemset(d_density_init_, 0, sizeof(float) * num_));
    h_density_init_ = new float[num_];
    memset(h_density_init_, 0, sizeof(float) * num_);

    h_is_boundary_ = new bool[num_];
    memset(h_is_boundary_, 0, sizeof(bool) * num_);
    CUDA_SAFE_CALL(cudaMalloc(&d_is_boundary_, sizeof(bool) * num_));
    CUDA_SAFE_CALL(cudaMemset(d_is_boundary_, 0, sizeof(bool) * num_));
}

void Particles::AddPlaneBoundary(float3 start, float3 end, float d, std::string axis) {
    // uniformly sample points on the plane
    std::vector<float3> boundary_points;
    if (axis == "x") {
        if (start.x != end.x || start.y > end.y || start.z > end.z) {
            std::cerr << "Error: given boundary points are invalid" << std::endl;
            exit(1);
        }
        if (start.x > bound_min_.x && start.x < bound_max_.x) {
            std::cerr << "Error: given boundary crosses the bounding box" << std::endl;
        }
        // exclude the start and end points
        for (float z = start.z + d / 2.0f; z < end.z; z += d) {
            for (float y = start.y + d / 2.0f; y < end.y; y += d) {
                boundary_points.push_back(make_float3(start.x, y, z));
            }
        }
    } else if (axis == "y") {
        if (start.y != end.y || start.x > end.x || start.z > end.z) {
            std::cerr << "Error: given boundary points are invalid" << std::endl;
            exit(1);
        }
        if (start.y > bound_min_.y && start.y < bound_max_.y) {
            std::cerr << "Error: given boundary crosses the bounding box" << std::endl;
        }
        // exclude the start and end points
        for (float z = start.z + d / 2.0f; z < end.z; z += d) {
            for (float x = start.x + d / 2.0f; x < end.x; x += d) {
                boundary_points.push_back(make_float3(x, start.y, z));
            }
        }
    } else if (axis == "z") {
        if (start.z != end.z || start.x > end.x || start.y > end.y) {
            std::cerr << "Error: given boundary points are invalid" << std::endl;
            exit(1);
        }
        if (start.z > bound_min_.z && start.z < bound_max_.z) {
            std::cerr << "Error: given boundary crosses the bounding box" << std::endl;
        }
        // exclude the start and end points
        for (float y = start.y + d / 2.0f; y < end.y; y += d) {
            for (float x = start.x + d / 2.0f; x < end.x; x += d) {
                boundary_points.push_back(make_float3(x, y, start.z));
            }
        }
    } else {
        std::cerr << "Error: invalid axis " << axis << std::endl;
        exit(1);
    }

    // add boundary points to the particle system
    int num_boundary_points = boundary_points.size();
    bool *h_is_boundary_new = new bool[num_ + num_boundary_points];
    memset(h_is_boundary_new, true, sizeof(bool) * (num_ + num_boundary_points));
    float3 *h_pos_new = new float3[num_ + num_boundary_points];
    float3 *h_vel_new = new float3[num_ + num_boundary_points];
    memset(h_vel_new, 0, sizeof(float3) * (num_ + num_boundary_points));
    float *h_density_init_new = new float[num_ + num_boundary_points];
    memset(h_density_init_new, 0, sizeof(float) * (num_ + num_boundary_points));
    CUDA_SAFE_CALL(cudaMemcpy(h_is_boundary_new, h_is_boundary_, sizeof(bool) * num_, cudaMemcpyHostToHost));
    CUDA_SAFE_CALL(cudaMemcpy(h_pos_new, h_pos_, sizeof(float3) * num_, cudaMemcpyHostToHost));
    CUDA_SAFE_CALL(cudaMemcpy(h_vel_new, h_vel_, sizeof(float3) * num_, cudaMemcpyHostToHost));
    CUDA_SAFE_CALL(cudaMemcpy(h_density_init_new, h_density_init_, sizeof(float) * num_, cudaMemcpyHostToHost));
    CUDA_SAFE_CALL(cudaMemcpy(h_pos_new + num_, boundary_points.data(), sizeof(float3) * num_boundary_points, cudaMemcpyHostToHost));
    delete[] h_pos_;
    delete[] h_is_boundary_;
    delete[] h_vel_;
    delete[] h_density_init_;
    h_pos_ = h_pos_new;
    h_is_boundary_ = h_is_boundary_new;
    h_vel_ = h_vel_new;
    h_density_init_ = h_density_init_new;
    num_ += num_boundary_points;
    CUDA_SAFE_CALL(cudaFree(d_pos_));
    CUDA_SAFE_CALL(cudaFree(d_is_boundary_));
    CUDA_SAFE_CALL(cudaFree(d_vel_));
    CUDA_SAFE_CALL(cudaFree(d_density_init_));
    CUDA_SAFE_CALL(cudaMalloc(&d_pos_, sizeof(float3) * num_));
    CUDA_SAFE_CALL(cudaMalloc(&d_is_boundary_, sizeof(bool) * num_));
    CUDA_SAFE_CALL(cudaMalloc(&d_vel_, sizeof(float3) * num_));
    CUDA_SAFE_CALL(cudaMalloc(&d_density_init_, sizeof(float) * num_));
    CUDA_SAFE_CALL(cudaMemcpy(d_pos_, h_pos_, sizeof(float3) * num_, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_is_boundary_, h_is_boundary_, sizeof(bool) * num_, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_vel_, h_vel_, sizeof(float3) * num_, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_density_init_, h_density_init_, sizeof(float) * num_, cudaMemcpyHostToDevice));

    // update bounding box
    bound_min_ = make_float3(std::min(bound_min_.x, start.x), std::min(bound_min_.y, start.y), std::min(bound_min_.z, start.z));
    bound_max_ = make_float3(std::max(bound_max_.x, end.x), std::max(bound_max_.y, end.y), std::max(bound_max_.z, end.z));
}

void Particles::OutputToOBJ(const std::string &filename) const {
    std::ofstream fout(filename);
    if (!fout) {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        exit(1);
    }
    fout << "# Happy Go Lucky" << std::endl;

    for (int i = 0; i < num_; ++i) {
        if (!h_is_boundary_[i])
            fout << "v " << h_pos_[i].x << " " << h_pos_[i].y << " " << h_pos_[i].z << std::endl;
    }

    fout.close();
}

void Particles::OutputToVTK(const std::string &filename, float *scalar) const {
    std::ofstream fout(filename);
    if (!fout) {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        exit(1);
    }
    fout << "# vtk DataFile Version 3.0" << std::endl;
    fout << "Happy Go Lucky" << std::endl;
    fout << "ASCII" << std::endl;
    fout << "DATASET POLYDATA" << std::endl;
    fout << "POINTS " << model_num_ << " float" << std::endl;
    for (int i = 0; i < num_; ++i) {
        if (!h_is_boundary_[i])
            fout << h_pos_[i].x << " " << h_pos_[i].y << " " << h_pos_[i].z << std::endl;
    }
    if (scalar != nullptr) {
        fout << "POINT_DATA " << model_num_ << std::endl;
        fout << "SCALARS scalar float" << std::endl;
        fout << "LOOKUP_TABLE default" << std::endl;
        for (int i = 0; i < num_; ++i) {
            if (!h_is_boundary_[i])
                fout << scalar[i] << std::endl;
        }
    }
}

void Particles::InitDensity(float *d_density) {
    CUDA_SAFE_CALL(cudaMemcpy(d_density_init_, d_density, sizeof(float) * num_, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(h_density_init_, d_density, sizeof(float) * num_, cudaMemcpyDeviceToHost));
}

void Particles::Reorder(int *h_index) {
    float3 *h_pos_new = new float3[num_];
    float3 *h_vel_new = new float3[num_];
    float *h_density_new = new float[num_];
    bool *h_is_boundary_new = new bool[num_];
    for (int i = 0; i < num_; ++i) {
        int new_pos = h_index[i];
        assert(new_pos < num_ && new_pos >= 0);
        h_pos_new[new_pos] = h_pos_[i];
        h_vel_new[new_pos] = h_vel_[i];
        h_density_new[new_pos] = h_density_init_[i];
        h_is_boundary_new[new_pos] = h_is_boundary_[i];
    }
    delete[] h_pos_;
    delete[] h_vel_;
    delete[] h_density_init_;
    delete[] h_is_boundary_;
    h_pos_ = h_pos_new;
    h_vel_ = h_vel_new;
    h_density_init_ = h_density_new;
    h_is_boundary_ = h_is_boundary_new;
    CUDA_SAFE_CALL(cudaMemcpy(d_pos_, h_pos_, sizeof(float3) * num_, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_vel_, h_vel_, sizeof(float3) * num_, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_density_init_, h_density_init_, sizeof(float) * num_, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_is_boundary_, h_is_boundary_, sizeof(bool) * num_, cudaMemcpyHostToDevice));
}

__global__ void knUpdateX(float3 *pos, float3 *vel, bool *is_boundary, float dt, int num) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num)
        return;
    if (!is_boundary[i])
        pos[i] = make_float3(pos[i].x + vel[i].x * dt, pos[i].y + vel[i].y * dt, pos[i].z + vel[i].z * dt);
}

void Particles::UpdateX(float dt) {
    int block_size = 256;
    int grid_size = GRID_SIZE(num_, block_size);
    knUpdateX<<<grid_size, block_size>>>(d_pos_, d_vel_, d_is_boundary_, dt, num_);
    CUDA_SAFE_CALL(cudaMemcpy(h_pos_, d_pos_, sizeof(float3) * num_, cudaMemcpyDeviceToHost));

    // update bounding box
    float x_min, y_min, z_min;
    x_min = y_min = z_min = std::numeric_limits<float>::max();
    float x_max, y_max, z_max;
    x_max = y_max = z_max = std::numeric_limits<float>::min();
    for (int i = 0; i < num_; ++i) {
        x_min = std::min(x_min, h_pos_[i].x);
        y_min = std::min(y_min, h_pos_[i].y);
        z_min = std::min(z_min, h_pos_[i].z);
        x_max = std::max(x_max, h_pos_[i].x);
        y_max = std::max(y_max, h_pos_[i].y);
        z_max = std::max(z_max, h_pos_[i].z);
    }
    bound_min_ = make_float3(x_min, y_min, z_min);
    bound_max_ = make_float3(x_max, y_max, z_max);
}

int3 Particles::GetCellResolution(float cell_size) {
    int3 res;
    res.x = (int)ceil((bound_max_.x - bound_min_.x) / cell_size);
    res.y = (int)ceil((bound_max_.y - bound_min_.y) / cell_size);
    res.z = (int)ceil((bound_max_.z - bound_min_.z) / cell_size);
    return res;
}

} // namespace xuan