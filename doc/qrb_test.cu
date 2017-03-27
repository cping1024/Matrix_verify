#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "glog/logging.h"
#include <sys/time.h>
static double getTimeOfMSeconds() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec*1000. + tv.tv_usec/1000.;
}

void  init_array(float *a, const int N);
void  init_mat(float *a, const int N, const int M);
float diff_array(float* a, float* b, const int N);
float diff_mat(float* a, float* b, const int N, const int M);
void  showHelp(const int argc, const char** argv);
void  runTest(int argc, const char* argv);

int main(int argc, const char* argv){
    bool bTestResult = false;

    /// start the log
    fprintf(stderr, );
}


int main(int argc, char* argv[]) {

        cublasStatus_t stat;
        cublasHandle_t handle;
        stat = cublasCreate(&handle);
        if (CUBLAS_STATUS_SUCCESS != stat) {
                std::cout << "cublasCreate Error." << std::endl;
                return 0;
        }

        const unsigned int M = 10000000;
        const unsigned int N = 160;
        float totalsz = (float)(M * (N + 1)) / 1204;
        totalsz *= 4; totalsz /= 1024; totalsz /=1024;
        fprintf(stderr, "Need tatal %f GB memory\n", totalsz);

        float* deviceMatrix;
        cudaMalloc((void**)&deviceMatrix, sizeof(float)*M*N);
        float* deviceVector;
        cudaMalloc((void**)&deviceVector, sizeof(float)*N);

        float* hostMatrix = new float[M*N];
        for (unsigned int i = 0; i < M*N; ++i) {
                hostMatrix[i] = 1.0 * rand() / RAND_MAX;
        }
        float* hostVector = new float[N];
        for (unsigned int i = 0; i < N; ++i) {
                hostVector[i] = 1.0* rand() / RAND_MAX;
        }
        cudaMemcpy(deviceMatrix, hostMatrix, sizeof(float)*M*N, cudaMemcpyHostToDevice);        

        float* deviceOut;
        cudaMalloc((void**)&deviceOut, sizeof(float)*M);
        cudaMemset(deviceOut, 0, sizeof(float)*M);

        cublasOperation_t trans = CUBLAS_OP_N;
        float alpha = 1.0;
        float beta = 0.0;
        double st = getTimeOfMSeconds();
        std::cout << "start sgemv\n";

        cudaMemcpy(deviceVector, hostVector, sizeof(float)*N, cudaMemcpyHostToDevice);

        stat = cublasSgemv(handle, trans, M, N, &alpha, deviceMatrix, M, deviceVector, 1, &beta, deviceOut, 1);
        cudaDeviceSynchronize();
        double et = getTimeOfMSeconds();
        std::cout << "cublasSgemv time " << et-st << "ms" << std::endl;

        thrust::counting_iterator<int> iter(0);
        thrust::device_vector<int> indices(M);
        thrust::copy(iter, iter+indices.size(), indices.begin());
        thrust::device_ptr<float> keys( deviceOut );

        st = getTimeOfMSeconds();
        thrust::sort_by_key(keys, keys+M, indices.begin(), thrust::greater<float>());
        cudaDeviceSynchronize();
        et = getTimeOfMSeconds();
        std::cout << "sort_by_key use time " << et-st << "ms" << std::endl;

        cudaFree(deviceOut);
        cudaFree(deviceVector);
        cudaFree(deviceMatrix);

        delete[] hostVector;
        delete[] hostMatrix;
        return 0;
}

