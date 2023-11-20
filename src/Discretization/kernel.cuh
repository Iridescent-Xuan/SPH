#ifndef __KERNEL_H__
#define __KERNEL_H__

#include <Utilities/util.h>
#include <Utilities/cuda_safe_call.h>

namespace xuan {

// Base class for all kernel functions
class CubicSplineKernel {
public:
    CubicSplineKernel(float h) : h_(h) {
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_h_, sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpy(d_h_, &h_, sizeof(float), cudaMemcpyHostToDevice));

        sigma_ = 8.0f / (MY_PI * h * h * h);
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_sigma_, sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpy(d_sigma_, &sigma_, sizeof(float), cudaMemcpyHostToDevice));
    }

    virtual ~CubicSplineKernel() {
        CUDA_SAFE_CALL(cudaFree(d_h_));
        CUDA_SAFE_CALL(cudaFree(d_sigma_));
    }

    __device__ float Q(float3 r) {
        float r_norm = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
        return r_norm / *d_h_;
    }

    __device__ float3 DqDxi(float3 r) {
        float r_norm = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
        float3 n = r_norm < 1e-6f ? make_float3(0.0, 0.0, 0.0) : make_float3(r.x / r_norm, r.y / r_norm, r.z / r_norm);
        return make_float3(n.x / *d_h_, n.y / *d_h_, n.z / *d_h_);
    }

    __device__ float Compute(float3 r) {
        float q = this->Q(r);
        if (q >= 0 && q <= 0.5f) {
            return *d_sigma_ * (6.0f * (q * q * q - q * q) + 1.0f);
        } else if (q > 0.5f && q <= 1.0f) {
            return *d_sigma_ * 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
        } else {
            return 0.0f;
        }
    }

    __device__ float3 Gradient(float3 r) { // w.r.t. x_i
        float grad;
        float q = this->Q(r);
        if (q >= 0 && q <= 0.5f) {
            grad = *d_sigma_ * (18.0f * q * q - 12.0f * q);
        } else if (q > 0.5f && q <= 1.0f) {
            grad = *d_sigma_ * (-6.0f) * (1.0f - q) * (1.0f - q);
        } else {
            grad = 0.0f;
        }
        float3 dqdxi = this->DqDxi(r);
        return make_float3(grad * dqdxi.x, grad * dqdxi.y, grad * dqdxi.z);
    }

private:
    float h_;        // smoothing length
    float *d_h_;     // on the device
    float sigma_;    // coefficient
    float *d_sigma_; // on the device
};

} // namespace xuan

#endif // __KERNEL_H__