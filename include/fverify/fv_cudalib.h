#ifndef _FV_CUDA_LIB_H_
#define _FV_CUDA_LIB_H_

#include "fv_define.h"
#include <cuda_runtime.h>

namespace FEATURE_VERIFYNS{

__constant__ float g_x[5];
__constant__ float g_k[4];
__constant__ float g_b[4];

class fvGPUKernel{
public:
    static bool AddFeature(float* matrix, float* feature, int64_t id, int64_t rows, int64_t cols);
    static bool VectorAdd(float* C, float* A, float* B, int64_t nelement);
    static bool ScoreMap(float* score, int64_t M);
};
} /// end namespace FEATURE_VERIFYNS



#endif
