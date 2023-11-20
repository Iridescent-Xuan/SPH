#ifndef _CUDA_SAFE_CALL_H_
#define _CUDA_SAFE_CALL_H_

#include <cuda_runtime.h>
#include <stdio.h>

// This macro is used to check return values from CUDA runtime calls
// [https://stackoverflow.com/a/14038590/20988759]

#define CUDA_SAFE_CALL(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#endif // _CUDA_SAFE_CALL_H_