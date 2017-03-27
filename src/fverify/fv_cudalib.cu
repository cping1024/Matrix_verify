#include "fv_cudalib.h"

#include "glog/logging.h"

using namespace FEATURE_VERIFYNS;

/// add feature to matrix with transpose mode
__global__ void add_featureKernel(float* matrix, float* feature, int64_t id, int64_t rows, int64_t cols){
    int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < cols){
        float val = feature[tid];
        int64_t pos = rows * tid + id;

        matrix[pos] = val;
    }
}
/// vector add method C = A + B, with nelement
__global__ void vector_addKernel(float* C, float* A, float* B, int64_t nelement){
    int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < nelement){
        float val = A[tid] + B[tid];
        C[tid] = val;
    }
}
__global__ void scoreMapKernel(float *score, int64_t M)
{
    int64_t id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < M){
        float m_score = score[id];

        /*for(int i = 1; i < 5; i++){
            if(m_score <= g_x[i]){
                m_score = m_score * g_k[i - 1] + g_b[i - 1];
                break;
            }
        }*/

        if(     m_score <= X1)m_score = m_score * K0 + B0;
        else if(m_score <= X2)m_score = m_score * K1 + B1;
        else if(m_score <= X3)m_score = m_score * K2 + B2;
        else if(m_score <= X4)m_score = m_score * K3 + B3;

        score[id] = (m_score <= 1 ? m_score : 1) * 100.0;
    }
}
bool fvGPUKernel::AddFeature(float* matrix, float* feature, int64_t id, int64_t rows, int64_t cols){
    int nThreads = 256;
    int nBlocks  = (cols + nThreads - 1) / nThreads;

    add_featureKernel<<<nBlocks, nThreads>>>(  matrix
                                             , feature
                                             , id
                                             , rows
                                             , cols);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
        LOG(FATAL) << "Failed to launch add feature kernel (error code: "
                   << cudaGetErrorString(err)
                   << ")";

    cudaDeviceSynchronize();
    return true;
}
bool fvGPUKernel::VectorAdd(float* C, float* A, float* B, int64_t nelement){
    int nThreads = 256;
    int nBlocks  = (nelement + nThreads - 1) / nThreads;

    vector_addKernel<<<nBlocks, nThreads>>>( C
                                           , A
                                           , B
                                           , nelement);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
        LOG(FATAL) << "Failed to launch VectorAdd kernel (error code: "
                   << cudaGetErrorString(err)
                   << ")";

    cudaDeviceSynchronize();
    return true;
}
bool fvGPUKernel::ScoreMap(float* score, int64_t M){
    int nThreads = 256;
    int nBlocks  = (M + nThreads - 1) / nThreads;

    scoreMapKernel<<<nBlocks, nThreads>>>(score, M);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
        LOG(FATAL) << "Failed to launch score map kernel (error code: "
                   << cudaGetErrorString(err)
                   << ")";

    cudaDeviceSynchronize();
    return true;
}
