#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include <algorithm>
#include <omp.h>

#include <helper_functions.h>
#include <helper_cuda.h>

/// sensetime face sdk version 5.2.0
/// here test titanz block size = 65535

#ifndef STFACE_VER_520
#define STFACE_VER_520
#endif

#ifndef int64_t
#define int64_t long long int
#endif

#ifndef nullptr
#define nullptr 0
#endif

#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

#ifndef MAXLIBLEN
#define MAXLIBLEN   15000000
#endif

#define outErr(FMT,ARG...)\
do{\
    char str[1024];\
    sprintf(str, "!!!ERR(File:[%s] Func:[%s] Line:[%d]):"FMT, __FILE__, __FUNCTION__, __LINE__, ##ARG);\
    fprintf(stderr, "%s\n", str);\
}while(0)

#define outMsg(FMT,ARG...)\
do{\
    char str[1024];\
    sprintf(str, "Info: "FMT, ##ARG);\
    fprintf(stderr, "%s\n", str);\
}while(0)

#ifndef CLAMP
#define CLAMP(a, min, max) ( MIN(max, MAX(a, min)) )
#endif

/// senseTime score map
#ifdef STFACE_VER_472
    float x[5] = { -1, 0.39, 0.44, 0.5, 1 };
    float y[5] = { 0, 0.5, 0.7, 0.9, 1 };
#endif //STFACE_VER_472

#ifdef STFACE_VER_520
    float x[5] = { -1, 0.43, 0.5, 0.55, 1 };
    float y[5] = { 0, 0.5, 0.7, 0.9, 1 };
#endif //STFACE_VER_520

float k[4];
float b[4];

__constant__ float g_x[5];
__constant__ float g_k[4];
__constant__ float g_b[4];
__constant__ float g_val[160];

/// cublas define
cublasHandle_t cublas_handle;
float alpha = 1.0;
float beta = 0.0;

double getTimeOfMSeconds();
void  init_array(float *a, int64_t N);
void  init_mat(float *a, int64_t N, int64_t M);
void  init_scoreMap();
bool  doGPUTranspose();
bool  add_feature(float* g_matrix, float* c_vector, float* g_vector, int64_t id, int64_t M, int64_t N);
bool  gpuRawCalc(float* c_vector, float* g_vector, float* g_matrix, float* g_output, int64_t M, int64_t N);
bool  cpuRawCalc(float* c_vector, float* c_matrix, float* c_output, int64_t M, int64_t N);
bool  gpuBlas(float* c_vector, float* g_vector
            , float* g_matrix, float* g_output
            , int64_t M, int64_t N);
int64_t gpuLibPick(float* g_score, int* g_index, int* g_libid, float* g_picklib, int64_t M, int libid);
bool  gpuThrust(float* g_score, int* g_index, int* g_sortedID, int64_t M);
bool  diff_array(float* a, float* b, int64_t N, const float tolerance);
bool  diff_mat(float* a, float* b, int64_t N, int64_t M, const float tolerance);
void  showHelp(const int argc, const char** argv);
bool  doGPUScoreMap(float* g_score, int64_t M);
bool  runTest(int argc, const char** argv);

int main(int argc, const char** argv){
    bool bTestResult = false;

    /// start the log
    outMsg("%s Starting ... \n", argv[0]);

    if (checkCmdLineFlag(argc, (const char **)argv, "help")){
        outMsg("Display help on console\n");

        showHelp(argc, (const char **)argv);
        bTestResult = true;
    } else {
        bTestResult = runTest(argc, (const char **)argv);
    }

    return bTestResult ? 0 : -1;
}
__global__ void scoreMap(float *score, int64_t M)
{
    int64_t id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < M){
        float m_score = score[id];

        #pragma unroll 4
        for(int i = 1; i < 5; i++){
            if(m_score <= g_x[i]){
                m_score = m_score * g_k[i - 1] + g_b[i - 1];
                break;
            }
        }
        score[id] = (m_score <= 1 ? m_score : 1) * 100.0;
    }
}
__global__ void rawCalc(float *matrix, float* output, int64_t M, int64_t N)
{
    int64_t id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < M){
        float m_score;
        float* in = matrix + id * N;

        for(int i = 0; i < N; i++){
            m_score += in[i] * g_val[i];
        }
        output[id] = m_score;
    }
}
__global__ void add_featureKernel(float* matrix, float* feature, int64_t id, int64_t M, int64_t N){
    int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < N){
        float val = feature[tid];
        int64_t pos = M * tid + id;

        matrix[pos] = val;
    }
}
__global__ void vector_addKernel(float* v1, float* v2, int64_t N){
    int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < N){
        float val = v1[tid] + v2[tid];
        v1[tid] = val;
    }
}
void  init_array(float *a, int64_t N){
    assert(a != nullptr);

//#pragma omp parallel for
    for(int64_t i = 0; i < N; i++){
        //unsigned int seed = 1.0;
        //a[i] = 1.0 * rand_r(&seed) / RAND_MAX;
        a[i] = 1.0 * rand() / RAND_MAX;
    }
}
void  init_mat(float *a, int64_t N, int64_t M){
    init_array(a, M * N);
}
bool diff_array(float* a, float* b, int64_t N, const float tolerance){
    assert(a != nullptr);
    assert(b != nullptr);

    double totalerr = 0.0;
    double totalval = 0.0;
    for(int64_t i = 0; i < N; i++){
        float diff = fabs(a[i] - b[i]);
        float error;

        if(a[i] != 0)
            error = diff / a[i];
        else
            error = diff;

        if(error > tolerance){
            outMsg("Data error at point (%lld)\t%f instead of %f\n", i, a[i], b[i]);
            return false;
        }
        totalerr += error * error;
        totalval += fabs(a[i]);
    }
    outMsg("Total value [%f] average [%f] error measure [%f]."
           , totalval, totalval / N, totalerr);
    return true;
}
bool diff_mat(float* a, float* b, int64_t N, int64_t M, const float tolerance){
    assert(a != nullptr);
    assert(b != nullptr);

    for(int64_t i = 0; i < M; i++){
        for(int64_t j = 0; j < N; j++){
            float diff = fabs(*a - *b);
            float error;

            if(*a != 0)error = diff / *a;
            else
                error = diff;

            if(error > tolerance){
                outMsg("Data error at point (%lld, %lld)\t%f instead of %f\n", i, j, *a, *b);
                return false;
            }

            ++a;
            ++b;
        }
    }

    return true;
}
void  showHelp(const int argc, const char** argv){
    if (argc > 0)
        std::cout << std::endl << argv[0] << std::endl;

    std::cout << std::endl << "Syntax:" << std::endl;
    std::cout << std::left;
    std::cout << "    " << "--row=<N>" << "Specify number of feature, default = 10000" << std::endl;
    std::cout << "    " << "--col=<N>" << "Specify feature dimentions, default = 160" << std::endl;
    std::cout << "    " << "--gpuid=<N>" << "Specify device id, default = 0" << std::endl;
    std::cout << "    " << "--compare=[0, 1]" << "Compare cpu result with gpu result, default = 0" << std::endl;
    std::cout << std::endl;
}
bool  doGPUTranspose(){
#define tFREE(){\
    if(c_matrix)free(c_matrix);   c_matrix = nullptr;\
    if(c_trans)free(c_trans); c_trans = nullptr;\
    if(c_vector)free(c_vector); c_vector = nullptr;\
    if(c_changed)free(c_changed); c_changed = nullptr;\
\
    if(g_matrix)cudaFree(g_matrix);   g_matrix = nullptr;\
    if(g_trans)cudaFree(g_trans); g_trans = nullptr;\
    if(g_trans2)cudaFree(g_trans2); g_trans2 = nullptr;\
    if(g_vector)cudaFree(g_vector); g_vector = nullptr;\
}
#define tallocCPU(ptr, size, type) {\
    if((ptr = (type *)malloc(size * sizeof(type))) == NULL){\
        outMsg("No more CPU memory for computer!");\
        tFREE();\
        return false;\
    }\
}
#define tallocGPU(ptr, size, type) {\
    if((cudaMalloc((void**)&ptr, size * sizeof(type))) != cudaSuccess){\
        outMsg("No more GPU memory for computer!");\
        tFREE();\
        return false;\
    }\
}

    float* c_matrix = nullptr;
    float* g_matrix = nullptr;
    float* c_trans = nullptr;
    float* g_trans = nullptr;
    float* g_trans2 = nullptr;
    float* c_vector = nullptr;
    float* g_vector = nullptr;

    float* c_changed = nullptr;

    int M = 65535;
    int N = 160;

    /// alloc cpu buffer
    tallocCPU(c_matrix, M * N, float);
    tallocGPU(g_matrix, M * N, float);
    tallocCPU(c_trans,  M * N, float);
    tallocGPU(g_trans,  M * N, float);
    tallocGPU(g_trans2, M * N, float);
    tallocCPU(c_vector, N, float);
    tallocGPU(g_vector, N, float);

    tallocCPU(c_changed, 200 * N, float);

    /// init cublas
    if (CUBLAS_STATUS_SUCCESS != cublasCreate(&cublas_handle)) {
        outMsg("Init cublas failed?");
        tFREE();
        return false;
    }

    /// set matrix value
    init_array(c_matrix, M * N);

    memcpy(c_trans, c_matrix, sizeof(float) * M * N);
    memset(c_trans + (M - 200) * N, 0, sizeof(float) * 200 * N);
    double startT = getTimeOfMSeconds();
    /// cpy matrix to gpu
    if((cudaMemcpy(g_matrix, c_trans, sizeof(float) * N * M, cudaMemcpyHostToDevice)) != cudaSuccess){
        outMsg("Cpy matrix from host to device failed?");
        tFREE();
        return false;
    }
    //cudaMemset(g_matrix + (M - 200) * N, 0, sizeof(float) * 200 * N);

    double endT = getTimeOfMSeconds();
    outMsg("cpy matrix from cpy to gpu %f ms", endT - startT);

    float alpha = 1.;
    float beta  = 0.;
    /// do gpu trans test
    startT = getTimeOfMSeconds();
    if(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, &alpha, g_matrix, N, &beta, g_matrix, N, g_trans, M)
            != CUBLAS_STATUS_SUCCESS){
        outMsg("cublas sgemv run failed?");
        return false;
    }
    cudaDeviceSynchronize();

    endT = getTimeOfMSeconds();
    outMsg("gpu matrix trans with cublas %f ms", endT - startT);
    cudaMemcpy(g_matrix, g_trans, sizeof(float) * M * N, cudaMemcpyDeviceToDevice);

    if(0){
        /// do cpu transform
        startT = getTimeOfMSeconds();
        for(int64_t ifeature = 0; ifeature < M; ifeature++){
            for(int64_t idim = 0; idim < N; idim++){
                int64_t out  = idim * M + ifeature;
                int64_t in   = ifeature * N + idim;
                c_trans[out] = c_matrix[in];
            }
        }
        endT = getTimeOfMSeconds();
        outMsg("cpu matrix trans %f ms", endT - startT);

        /// compare gpu to cpu
        if((cudaMemcpy(c_matrix, g_trans, sizeof(float) * N * M, cudaMemcpyDeviceToHost)) != cudaSuccess){
            outMsg("Cpy matrix from device to host failed?");
            tFREE();
            return false;
        }

        diff_array(c_matrix, c_trans, M * N, 1.0e-4);
    }

    /// try add multi feature with transpose
    memset(c_trans, 0, sizeof(float) * M * N);
    memcpy(c_trans + (M - 200) * N, c_matrix + (M - 200) * N, sizeof(float) * 200 * N);
    if((cudaMemcpy(g_trans, c_trans, sizeof(float) * N * M, cudaMemcpyHostToDevice)) != cudaSuccess){
        outMsg("Cpy matrix from host to device failed?");
        tFREE();
        return false;
    }

    /// transpose g_trans
    startT = getTimeOfMSeconds();
    if(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, &alpha, g_trans, N, &beta, g_trans, N, g_trans2, M)
            != CUBLAS_STATUS_SUCCESS){
        outMsg("cublas sgemv run failed?");
        return false;
    }
    cudaDeviceSynchronize();
    endT = getTimeOfMSeconds();
    outMsg("gpu matrix trans with cublas %f ms", endT - startT);

    /// add g_trans2 to g_matrix
    int nThreads = 256;
    int nBlocks  = (N * M + nThreads - 1) / nThreads;
    vector_addKernel<<<nBlocks, nThreads>>>(g_matrix, g_trans2, M * N);

    /// add to g_matrix
    /// do cpu transpose matrix to trans
    for(int64_t ifeature = 0; ifeature < M; ifeature++){
        for(int64_t idim = 0; idim < N; idim++){
            int64_t out  = idim * M + ifeature;
            int64_t in   = ifeature * N + idim;
            c_trans[out] = c_matrix[in];
        }
    }
    if((cudaMemcpy(c_matrix, g_matrix, sizeof(float) * N * M, cudaMemcpyDeviceToHost)) != cudaSuccess){
        outMsg("Cpy matrix from device to host failed?");
        tFREE();
        return false;
    }

    diff_array(c_matrix, c_trans, M * N, 1.0e-4);

    tFREE();
    return true;


#undef tFREE
#undef tallocCPU
#undef tallocGPU
}
bool  add_feature(float* g_matrix, float* c_vector, float* g_vector, int64_t id, int64_t M, int64_t N){
    outMsg("Start feature add operator ...");
    if(id < 0 || id >= M){
        outMsg("Bad feature id [%lld], which should between [0 - %lld]", id, M - 1);
        return false;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    /// copy c_vector 2 g_vector
    if((cudaMemcpy(g_vector, c_vector, sizeof(float) * N, cudaMemcpyHostToDevice)) != cudaSuccess){
        outMsg("Cpy vector from host to device failed?");
        return false;
    }

    /// reset feature to matrix, here need transpose
    int nThreads = 256;
    int nBlocks  = (N + nThreads - 1) / nThreads;

    add_featureKernel<<<nBlocks, nThreads>>>(g_matrix, g_vector, id, M, N);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        outMsg("Failed to launch Add_feature kernel (error code %s)!\n", cudaGetErrorString(err));
        return false;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    outMsg("Add_feature running time %f ms", milliseconds);

    return true;
}
void  init_scoreMap(){
    for(int i = 0; i < 4; i++){
        k[i] = (y[i] - y[i + 1]) / (x[i] - x[i + 1]);
        b[i] = y[i] - k[i] * x[i];
        /// outMsg("No.%d k = %f/tb = %f", i + 1, k[i], b[i]);
    }

    /// copy k/b/x to gpu
    cudaMemcpyToSymbol(g_x, x, sizeof(float) * 5);
    cudaMemcpyToSymbol(g_k, k, sizeof(float) * 4);
    cudaMemcpyToSymbol(g_b, b, sizeof(float) * 4);
}
bool  doGPUScoreMap(float* g_score, int64_t M){
    /// do score map based on k/b
    cudaError_t err = cudaSuccess;

    int nThreads = 256;
    int nBlocks  = (M + nThreads - 1) / nThreads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    scoreMap<<<nBlocks, nThreads>>>(g_score, M);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        outMsg("Failed to launch scoreMap kernel (error code %s)!\n", cudaGetErrorString(err));
        return false;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    outMsg("doGPUScoreMap running time %f ms", milliseconds);

    return true;
}
double getTimeOfMSeconds(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec*1000. + tv.tv_usec/1000.;
}
bool  gpuRawCalc(float* c_vector, float* g_vector, float* g_matrix, float* g_output, int64_t M, int64_t N){
    outMsg("Start GPU raw calc ...");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaError_t err = cudaSuccess;
    err = cudaMemcpyToSymbol(g_val, c_vector, sizeof(float) * N);
    if (err != cudaSuccess)
    {
        outMsg("Failed to cudaMemcpyToSymbol (error code %s)!\n", cudaGetErrorString(err));
        return false;
    }

    int nThreads = 256;
    int nBlocks  = (M + nThreads - 1) / nThreads;

    rawCalc<<<nBlocks, nThreads>>>(g_matrix, g_output, M, N);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        outMsg("Failed to launch rawCalc kernel (error code %s)!\n", cudaGetErrorString(err));
        return false;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    outMsg("gpuRawCalc running time %f ms", milliseconds);

    return true;
}
bool  gpuBlas(float* c_vector, float* g_vector
            , float* g_matrix, float* g_output
            , int64_t M, int64_t N){
    outMsg("Start GPU blas computer ...");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    /// cpu vector 2 gpu vector
    if((cudaMemcpy(g_vector, c_vector, sizeof(float) * N, cudaMemcpyHostToDevice)) != cudaSuccess){
        outMsg("Cpy vector from host to device failed?");
        return false;
    }

    /// init output value
    if(cudaMemset(g_output, 0, sizeof(float) * M) != cudaSuccess){
        outMsg("gpu_score memset zero failed?");
        return false;
    }

    /// run cublas
    if(cublasSgemv(cublas_handle, CUBLAS_OP_N, M, N, &alpha, g_matrix, M, g_vector, 1, &beta, g_output, 1)
            != CUBLAS_STATUS_SUCCESS){
        outMsg("cublas sgemv run failed?");
        return false;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    outMsg("cublas running time %f ms", milliseconds);

    return true;
}
int64_t gpuLibPick(float* g_score, int* g_index, int* g_libid, float* g_picklib, int64_t M, int libid){
    /// do libid pick
    /// int64_t nFeature = M;
    /// if(libid >= 0)nFeature = doLibPick();

    /// do score map based on senseTime SDK v5.2.0
    /// doGPUScoreMap(g_output, M);
    /// double scoremapT = getTimeOfMSeconds();


    return M;
}
bool  gpuThrust(float* g_score, int* g_index, int* g_sortedID, int64_t M){
    outMsg("Start GPU thrust computer ...");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    /// copy original index to sortedID
    if((cudaMemcpy(g_sortedID, g_index, sizeof(int) * M, cudaMemcpyDeviceToDevice)) != cudaSuccess){
        outMsg("Cpy index from recoder to sorted buffer failed?");
        return false;
    }
    /// sort result
    thrust::device_ptr<float> keys = thrust::device_pointer_cast(g_score);
    thrust::device_ptr<int>   vals = thrust::device_pointer_cast(g_sortedID);

    thrust::sort_by_key(keys, keys + M, vals, thrust::greater<float>());

   /*thrust::counting_iterator<int> iter(0);
    thrust::device_vector<int> indices(M);
    thrust::copy(iter, iter+indices.size(), indices.begin());
    thrust::device_ptr<float> keys( deviceOut );

    thrust::sort_by_key(keys, keys+M, indices.begin(), thrust::greater<float>());*/

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    outMsg("thrust running time %f ms", milliseconds);
    return true;
}
bool  cpuRawCalc(float* c_vector, float* c_matrix, float* c_output, int64_t M, int64_t N){
    outMsg("Start CPU raw computer ...");

    /*float* trans = (float*)malloc(M * N * sizeof(float));
    for(int64_t ifeature = 0; ifeature < M; ifeature++){
        for(int64_t idim = 0; idim < N; idim++){
            int64_t in  = idim * M + ifeature;
            int64_t out = ifeature * N + idim;
            trans[out] = c_matrix[in];
        }
    }*/
#pragma omp parallel for
    for(int64_t ifeature = 0; ifeature < M; ifeature++){
        float score = 0.0f;
        /// float* libptr = trans + ifeature * N;
        float* libptr = c_matrix + ifeature * N;
        for(int64_t idim = 0; idim < N; idim++){
            score += libptr[idim] * c_vector[idim];
        }

        /// do k/b map
        /*for(int i = 1; i < 5; i++){
            if(score < x[i]){
                score = score * k[i - 1] + b[i - 1];
                break;
            }
        }
        c_output[ifeature] = ((score <= 1) ? score : 1) * 100;*/
        c_output[ifeature] = score;
    }

    /// std::cout << "CPU raw calculate finished ..." << std::endl;

    /// free(trans);
    return true;
}
bool  runTest(int argc, const char** argv){

#define FREE(){\
    if(cpu_index)free(cpu_index);   cpu_index = nullptr;\
    if(cpu_score)free(cpu_score); cpu_score = nullptr;\
    if(cpu_sortedid)free(cpu_sortedid); cpu_sortedid = nullptr;\
    if(cpu_vector)free(cpu_vector); cpu_vector = nullptr;\
    if(cpu_matrix)free(cpu_matrix); cpu_matrix = nullptr;\
    if(cpu_libid)free(cpu_libid);   cpu_libid = nullptr;\
\
    if(gpu_index)cudaFree(gpu_index);   gpu_index = nullptr;\
    if(gpu_score)cudaFree(gpu_score); gpu_score = nullptr;\
    if(gpu_sortedid)cudaFree(gpu_sortedid); gpu_sortedid = nullptr;\
    if(gpu_vector)cudaFree(gpu_vector); gpu_vector = nullptr;\
    if(gpu_matrix)cudaFree(gpu_matrix); gpu_matrix = nullptr;\
    if(gpu_libid)cudaFree(gpu_libid);   gpu_libid = nullptr;\
    if(gpu_libpickscore)cudaFree(gpu_libpickscore); gpu_libpickscore = nullptr;\
    if(gpu_libpickindex)cudaFree(gpu_libpickindex); gpu_libpickindex = nullptr;\
}
#define allocCPU(ptr, size, type) {\
    if((ptr = (type *)malloc(size * sizeof(type))) == NULL){\
        outMsg("No more CPU memory for computer![%lld MB]", memlenCPU / 1024 / 1024);\
        FREE();\
        return false;\
    }\
    memlenCPU += size * sizeof(type);\
}
#define allocGPU(ptr, size, type) {\
    if((cudaMalloc((void**)&ptr, size * sizeof(type))) != cudaSuccess){\
        outMsg("No more GPU memory for computer![%lld MB]", memlenGPU / 1024 / 1024);\
        FREE();\
        return false;\
    }\
    memlenGPU += size * sizeof(type);\
}

    /// feature lib record index
    int* cpu_index  = nullptr;
    int* gpu_index  = nullptr;

    /// scores and index which sorted by score
    float* cpu_score = nullptr;
    float* gpu_score = nullptr;
    int*   cpu_sortedid = nullptr;
    int*   gpu_sortedid = nullptr;

    /// if specify lib id, for lib pick (score, index)
    float* gpu_libpickscore = nullptr;
    int*   gpu_libpickindex = nullptr;

    /// input feature which will be compared
    float* cpu_vector = nullptr;
    float* gpu_vector = nullptr;

    /// feature matrix
    float* cpu_matrix = nullptr;
    float* gpu_matrix = nullptr;

    /// feature lib id for each record
    int*   cpu_libid  = nullptr;
    int*   gpu_libid  = nullptr;

    int64_t M = 10000;
    int64_t N = 160;

    int64_t memlenCPU = 0;
    int64_t memlenGPU = 0;
    int gpuid = 0;
    int gpuN;
    int dotranspose = 0;

    bool compare = false;

    if (checkCmdLineFlag(argc, argv, "col"))
        N = CLAMP(getCmdLineArgumentInt(argc, argv, "col"), 64, 512);

    if (checkCmdLineFlag(argc, argv, "row"))
        M = CLAMP(getCmdLineArgumentInt(argc, argv, "row"), 10, MAXLIBLEN);

    if (checkCmdLineFlag(argc, argv, "compare"))
        compare = getCmdLineArgumentInt(argc, argv, "compare") == 0 ? false : true;

    if (checkCmdLineFlag(argc, argv, "gpuid"))
        gpuid = getCmdLineArgumentInt(argc, argv, "gpuid");

    if (checkCmdLineFlag(argc, argv, "dotranspose"))
        dotranspose = getCmdLineArgumentInt(argc, argv, "dotranspose");

    checkCudaErrors(cudaGetDeviceCount(&gpuN));
    if(gpuid < 0 || gpuid >= gpuN){
        outMsg("Bad parameter value gpuid = %d?", gpuid);
        return false;
    }
    checkCudaErrors(cudaSetDevice(gpuid));

    if(dotranspose == 1)return doGPUTranspose();

    double startT = getTimeOfMSeconds();
    /// malloc cpu buffer
    allocCPU(cpu_score, M, float);
    allocCPU(cpu_sortedid, M, int);
    allocCPU(cpu_index,  M, int);
    allocCPU(cpu_libid,  M, int);
    allocCPU(cpu_vector, N, float);
    allocCPU(cpu_matrix, M * N, float);

    double endT = getTimeOfMSeconds();
    outMsg("alloc cpu memory use time %f ms", endT - startT);

    ///omp_set_num_threads(16);
    startT = getTimeOfMSeconds();
    init_array(cpu_vector, N);
    init_mat(cpu_matrix, N, M);
    init_scoreMap();

#pragma omp parallel for
    for(int64_t i = 0; i < M; i++){
        cpu_index[i] = i;
        cpu_libid[i] = i % 5;
    }
    endT = getTimeOfMSeconds();
    outMsg("init cpu memory value use time %f ms", endT - startT);

    /// malloc gpu buffer
    startT = getTimeOfMSeconds();
    allocGPU(gpu_score, M, float);
    allocGPU(gpu_sortedid, M, int);
    allocGPU(gpu_libpickscore, M, float);
    allocGPU(gpu_libpickindex, M, int);
    allocGPU(gpu_index,  M, int);
    allocGPU(gpu_libid,  M, int);
    allocGPU(gpu_vector, N, float);
    allocGPU(gpu_matrix, M * N, float);
    endT = getTimeOfMSeconds();
    outMsg("alloc gpu memory value use time %f ms", endT - startT);

    outMsg("total %lld feature, cpu memory [%lld MB], gpu memory [%lld MB]."
           , M
           , memlenCPU / 1024 / 1024
           , memlenGPU / 1024 / 1024);

    startT = getTimeOfMSeconds();
    /// ===================================================================
    /// upload matrix && index
    /// here need transpose
    float* trans = (float*)malloc(M * N * sizeof(float));
    ///int64_t in = 0;
    for(int64_t ifeature = 0; ifeature < M; ifeature++){
        for(int64_t idim = 0; idim < N; idim++){
            int64_t out  = idim * M + ifeature;
            int64_t in   = ifeature * N + idim;
            trans[out] = cpu_matrix[in];
        }
    }

    if((cudaMemcpy(gpu_matrix, trans, sizeof(float) * M * N, cudaMemcpyHostToDevice)) != cudaSuccess){
        outMsg("Cpy matrix from host to device failed?");
        FREE();
        return false;
    }
    free(trans);
    endT = getTimeOfMSeconds();
    outMsg("trans + cpy matrix cpu data to gpu use time %f ms", endT - startT);
    /// ===================================================================
    if((cudaMemcpy(gpu_index, cpu_index, sizeof(int) * M, cudaMemcpyHostToDevice)) != cudaSuccess){
        outMsg("Cpy index from host to device failed?");
        FREE();
        return false;
    }
    if((cudaMemcpy(gpu_libid, cpu_libid, sizeof(int) * M, cudaMemcpyHostToDevice)) != cudaSuccess){
        outMsg("Cpy index from host to device failed?");
        FREE();
        return false;
    }

    endT = getTimeOfMSeconds();
    outMsg("copy cpu data to gpu use time %f ms", endT - startT);

    /// init cublas
    if (CUBLAS_STATUS_SUCCESS != cublasCreate(&cublas_handle)) {
        outMsg("Init cublas failed?");
        FREE();
        return false;
    }

    /// ===========================================================================
    /// add freature

    if(1){
        float* feature = new float[N];
        memset(feature, 0, sizeof(float) * N);
        if(!add_feature(gpu_matrix, feature, gpu_vector, M - 1, M, N)){
            FREE();
            delete []feature;
            return false;
        }

        delete feature;
    }
    /// ===========================================================================
    /// calculate gpu score
    if(!gpuBlas(cpu_vector, gpu_vector, gpu_matrix, gpu_score, M, N)){
        FREE();
        return false;
    }
    /*if(!gpuRawCalc(cpu_vector, gpu_vector, gpu_matrix, gpu_score, M, N)){
        FREE();
        return false;
    }*/

    /// ===========================================================================
    /// gpu score map

    /// do score maps
    /*startT = getTimeOfMSeconds();
    if(!doGPUScoreMap(gpu_score, M)){
        FREE();
        return false;
    }
    endT = getTimeOfMSeconds();
    outMsg("do gpu score map %f ms", endT - startT);*/

    /// ===========================================================================
    /// gpu score sort
    /// pick score based on libid
    /*int64_t nScore;
    if((nScore = gpuLibPick(gpu_score, gpu_index, gpu_libid, gpu_libpick, M, libid)) < 0){
        FREE();
        return false;
    }*/

    /// sort by key
    if(1)
    if(!gpuThrust(gpu_score, gpu_index, gpu_sortedid, M)){
        FREE();
        return false;
    }

    /// ====================================================================================
    /// CPU job
    startT = getTimeOfMSeconds();
    if(!cpuRawCalc(cpu_vector, cpu_matrix, cpu_score, M, N)){
        FREE();
        return false;
    }
    endT = getTimeOfMSeconds();
    outMsg("CPU raw calculate use time %f ms", endT - startT);

    /// compare cpu result with gpu result
    float* tmp = nullptr;
    allocCPU(tmp, M, float);
    if((cudaMemcpy(tmp, gpu_score, sizeof(float) * M, cudaMemcpyDeviceToHost)) != cudaSuccess){
        outMsg("Cpy matrix from device to host failed?");
        FREE();
        free(tmp);
        return false;
    }
    if(M > 2){
        float maxv = tmp[0], minv = tmp[0];
        for(int64_t id = 0; id < M; id++){
            if(tmp[id] > maxv)maxv = tmp[id];
            if(tmp[id] < minv)minv = tmp[id];
        }
        outMsg("Sorted value: max = %f, middle = %f, min = %f, get post max = %f, min = %f"
               , tmp[0], tmp[M / 2], tmp[M - 1]
               , maxv, minv);
    }

    /// sort cpu score
    std::sort(cpu_score, cpu_score + M, std::greater<float>());
    diff_array(tmp, cpu_score, M, 1.0e-4);
    free(tmp);

    FREE();

    return true;
}

