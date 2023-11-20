#ifndef __PARAMETER_H__
#define __PARAMETER_H__

#include <string>

namespace xuan {

struct SysParameter {
    float dt = 0.01f;       // time step
    float mass = 1.0f;      // particle mass
    float radius = 0.25f;   // particle radius
    float h = 0.1f;         // smoothing length
    int cell_res = 100;     // grid resolution
    float cell_size = 0.1f; // grid size, NOTE: should be equal to h
    float k_SE = 1e5;       // stiffness constant for state equation
    float3 gravity = make_float3(0.0f, -9.8f, 0.0f);
    float3 velocity = make_float3(0.0f, 0.0f, 0.0f);
    float boundary_sample_rate = 0.01f;
    std::string kernel = "cubic";

    void InitFromJson(const std::string &json_file) {
        // TODO:
    }
};

} // namespace xuan

#endif // __PARAMETER_H__