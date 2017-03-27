#include "fv_database.h"

#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include "fv_define.h"
#include "fv_cudalib.h"

using namespace FEATURE_VERIFYNS;
/// ==============================================
/// class FV_DBlocks
bool FV_DBlocks::AddFeature(int64_t nfeature, int64_t* fid, float* features){
    /// need optimize it later

    /// single feature method, too slow!
    /*for(int64_t i = 0; i < nfeature; i++){
        if(!AddFeature(fid[i], features + i * FV_FEATURESZ))
            return false;
    }

    return true;*/

    /// block method
    cublasStatus_t err;
    float alpha = 1.;
    float beta  = 0.;

    int64_t size = sizeof(float) * FV_BLOCKSZ * FV_FEATURESZ;
    int64_t nLeft = nfeature;
    int64_t nAdded = 0;

    do{
        FV_DBlock* ptr = NxtBlockptr();

        CHECK(ptr != NULL);
        CHECK_LE(ptr->Count(), FV_BLOCKSZ);

        int64_t  toAdd = MIN(FV_BLOCKSZ - ptr->Count(), nLeft);
        CHECK_GT(toAdd, 0);

        int64_t* fidptr = fid + nAdded;
        float*   ftptr  = features + nAdded * FV_FEATURESZ;

        /// ===========================
        /// add toAdd feature
        __TRYCUDA__(cudaMemset(d_swp_, 0, size * 2));
        __TRYCUDA__(cudaMemcpy(d_swp_ + ptr->Count() * FV_FEATURESZ
                             , ftptr
                             , sizeof(float) * toAdd * FV_FEATURESZ
                             , cudaMemcpyHostToDevice));

        /// do transpose
        err = cublasSgeam(cublas_handle[gpuid_]
                          , CUBLAS_OP_T
                          , CUBLAS_OP_T
                          , FV_BLOCKSZ
                          , FV_FEATURESZ
                          , &alpha
                          , d_swp_
                          , FV_FEATURESZ
                          , &beta
                          , d_swp_
                          , FV_FEATURESZ
                          , d_swp_ + FV_BLOCKSZ * FV_FEATURESZ
                          , FV_BLOCKSZ);

        CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Do add feature block failed!";
        cudaDeviceSynchronize();

        /// add to matrix
        CHECK(fvGPUKernel::VectorAdd(ptr->matrix_, ptr->matrix_, d_swp_ + FV_BLOCKSZ * FV_FEATURESZ, FV_BLOCKSZ * FV_FEATURESZ));

        /// add toAdd fid from fidptr
        __TRYCUDA__(cudaMemcpy(  ptr->fid_ + ptr->Count()
                               , fidptr
                               , sizeof(int64_t) * toAdd
                               , cudaMemcpyHostToDevice));

        /// adjust dblock count_
        ptr->count_ += toAdd;

        nLeft  -= toAdd;
        nAdded += toAdd;
    }while(nLeft > 0);

    return true;
}
bool FV_DBlocks::AddFeature(int64_t fid, float* feature){
    FV_DBlock* ptr = NxtBlockptr();

    CHECK(ptr != NULL);
    CHECK_LE(ptr->Count(), FV_BLOCKSZ);

    /// copy to d_swp
    __TRYCUDA__(cudaMemcpy(d_swp_, feature, sizeof(float) * FV_FEATURESZ, cudaMemcpyHostToDevice));

    /// submit kernel
    CHECK(fvGPUKernel::AddFeature(ptr->matrix_, d_swp_, ptr->Count(), FV_BLOCKSZ, FV_FEATURESZ));

    /// add fid
    __TRYCUDA__(cudaMemcpy(ptr->fid_ + ptr->Count(), &fid, sizeof(int64_t), cudaMemcpyHostToDevice));

    /// count++
    ptr->count_++;

    return true;
}
FV_DBlock* FV_DBlocks::NxtBlockptr(){
    FV_DBlock* ret = NULL;

    if(data_.size() > 0){
        ret = data_.at(data_.size() - 1);
        CHECK_LE(ret->Count(), FV_BLOCKSZ);
    }

    if(!ret || ret->Count() >= FV_BLOCKSZ){
        ret = new FV_DBlock();
        data_.push_back(ret);

        size_ += FV_DBlock::Size();
    }

    return ret;
}
int64_t FV_DBlocks::NumFeatures(){
    int64_t ret = 0;

    int size = data_.size();
    if(size == 0)return ret;
    ret = (size - 1) * FV_BLOCKSZ + data_.at(size - 1)->Count();

    return ret;
}
/// ==============================================
/// class FVDataBase
FVDataBase::FVDataBase(int gpuid, int64_t gpumemsz)
    : gpuid_(gpuid)
    , gpu_mem_sz_(gpumemsz)
    , h_scores_(NULL)
    , d_scores_(NULL)
    , h_fids_(NULL)
    , d_fids_(NULL){
    /// alloc swp area

    int64_t size = DSwpSize();
    if((h_matrixswp_ = (float*)malloc(size)) == NULL){
        LOG(FATAL) << "No enough memory for work! ["
                   << size / 1024
                   << " KB]";
    }

    __TRYCUDA__(cudaMalloc((void**)&d_matrixswp_, size));

    CHECK_GE(gpu_mem_sz_, 0);

    cublasStatus_t err = cublasCreate(&cublas_handle[gpuid_]);

    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Create cublas handle failed";

    /// init score map
    /// InitScoreMap();
}
FVDataBase::~FVDataBase(){
    int nlib = libs_.size();
    for(int i = 0; i < nlib; i++){
        FV_DBlocks* ptr = libs_.at(i);
        if(ptr)delete ptr;
    }

    libs_.clear();

    if(h_matrixswp_)free(h_matrixswp_); h_matrixswp_ = NULL;
    if(h_scores_)free(h_scores_); h_scores_ = NULL;
    if(h_fids_)free(h_fids_); h_fids_ = NULL;

    if(d_matrixswp_)__TRYCUDA__(cudaFree(d_matrixswp_));
    d_matrixswp_ = NULL;
    if(d_scores_)__TRYCUDA__(cudaFree(d_scores_));
    d_scores_ = NULL;
    if(d_fids_)__TRYCUDA__(cudaFree(d_fids_));
    d_fids_ = NULL;

    cublasStatus_t err = cublasDestroy(cublas_handle[gpuid_]);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Destory cublas handle failed";
}
int  FVDataBase::LibIndex(int libid){
    int size = libs_.size();
    for(int i = 0; i < size; i++){
        if(libs_.at(i) && (libid == libs_.at(i)->LibID()))
            return i;
    }

    return -1;
}
bool FVDataBase::ExistLib(int libid){
    return LibIndex(libid) >= 0;
}
FV_DBlocks* FVDataBase::Libsptr(int libid){
    int size = libs_.size();
    for(int i = 0; i < size; i++){
        if(libs_.at(i) && (libid == libs_.at(i)->LibID()))
            return libs_.at(i);
    }

    return NULL;
}
bool FVDataBase::AddLibDef(int libid){
    if(ExistLib(libid)){
        LOG(WARNING) << "Feature Lib id = "
                     << libid
                     << " already exist!";

        return false;
    }

    FV_DBlocks* newlib = new FV_DBlocks(gpuid_, libid, h_matrixswp_, d_matrixswp_);

    libs_.push_back(newlib);

    return true;
}
int64_t FVDataBase::NumFeatures(int libid){
    int64_t ret = 0;

    if(libid >= 0){
        int idx = LibIndex(libid);

        return idx == -1 ? ret : libs_.at(idx)->NumFeatures();
    }

    int size = libs_.size();
    for(int i = 0; i < size; i++){
        if(!libs_.at(i))continue;
        ret += libs_.at(i)->NumFeatures();
    }

    return ret;
}
int FVDataBase::NumBlock(int libid){
    int ret = 0;

    if(libid >= 0){
        int idx = LibIndex(libid);

        return idx == -1 ? ret : libs_.at(idx)->NumLibs();
    }

    int size = libs_.size();
    for(int i = 0; i < size; i++){
        if(!libs_.at(i))continue;
        ret += libs_.at(i)->NumLibs();
    }

    return ret;
}
int64_t FVDataBase::UsedMemSize(){
    /// swp buffer
    int64_t ret = DSwpSize();

    /// libs_
    int size = libs_.size();
    for(int i = 0; i < size; i++){
        if(!libs_.at(i))continue;
        ret += libs_.at(i)->Size();
    }

    /// score and sorted fids
    ret += (sizeof(float) + sizeof(int64_t)) * NumBlock() * FV_BLOCKSZ;

    return ret;
}
bool FVDataBase::AddFeature(int libid, int nfeature, int64_t* fid, float* features){
    int oblockN = NumBlock();

    if(!ExistLib(libid))AddLibDef(libid);

    FV_DBlocks* ptr = Libsptr(libid);
    CHECK(ptr != NULL) << "Weird things happend!";

    CHECK(ptr->AddFeature(nfeature, fid, features));

    int nblockN = NumBlock();
    CHECK_GE(nblockN, oblockN);

    if(nblockN != oblockN)return ReallocBuf();

    return true;
}
bool FVDataBase::AddFeature(int libid, int64_t fid, float* feature){
    int oblockN = NumBlock();
    if(!ExistLib(libid))AddLibDef(libid);

    FV_DBlocks* ptr = Libsptr(libid);
    CHECK(ptr != NULL) << "Weird things happend!";

    CHECK(ptr->AddFeature(fid, feature));

    int nblockN = NumBlock();
    CHECK_GE(nblockN, oblockN);

    if(nblockN != oblockN)return ReallocBuf();

    return true;
}
int64_t FVDataBase::NumFeatureCanbeAdded(int libid){
    int64_t ret = 0;

    /// calc new block num
    int64_t freeMem = gpu_mem_sz_ - UsedMemSize();
    int64_t unitBlocksz = FV_DBlock::Size();
                        //+ sizeof(float)   * FV_BLOCKSZ
                        //+ sizeof(int64_t) * FV_BLOCKSZ;

    int64_t freeBlocknum = freeMem / unitBlocksz;

    ret += freeBlocknum * FV_BLOCKSZ;

    /// add remaind buffer
    FV_DBlocks* ptr = Libsptr(libid);

    if(ptr){
        int remain = ptr->NumFeatures() % FV_BLOCKSZ;
        if(remain > 0)
            ret += FV_BLOCKSZ - remain;
    }

    return ret > 0 ? ret : 0;
}

int64_t FVDataBase::NumFeatureWanted(int libid){
    int64_t ret = 0;

    ret = NumFeatureCanbeAdded(libid);
    if(ret <= 0)return 0;

    FV_DBlocks* ptr = Libsptr(libid);

    if(!ptr)return FV_BLOCKSZ;

	ret =  ptr->NumFeatures() % FV_BLOCKSZ;
    return ret == 0 ? 0 : FV_BLOCKSZ - ret;
}
bool FVDataBase::FeatureVerify(float* ifeature, int libid, float* oscore, int64_t* ofids, int topnum){
    std::vector<int > libids;

    libids.push_back(libid);
    return FeatureVerify(ifeature, libids, oscore, ofids, topnum);
}
bool FVDataBase::FeatureVerify(float* ifeature, float* oscore, int64_t* ofids, int topnum){
    std::vector<int > libids;

    int size = libs_.size();
    for(int i = 0; i < size; i++)
        libids.push_back(libs_.at(i)->LibID());

    return FeatureVerify(ifeature, libids, oscore, ofids, topnum);
}
bool FVDataBase::FeatureVerify(float* ifeature, std::vector<int > libids, float* oscore, int64_t* ofids, int topnum){
    memset(oscore, 0, topnum * sizeof(float));
    memset(ofids,  0, topnum * sizeof(int64_t));

    int size = libids.size();
    if(size == 0)return true;

    /// LOG(INFO) << "No." << gpuid_ << "will do libids = " << libids.at(0) << " feature verify ...";
    /// copy ifeature to gpu
    __TRYCUDA__(cudaMemcpy(d_matrixswp_, ifeature, sizeof(float) * FV_FEATURESZ, cudaMemcpyHostToDevice));

    float* scoredptr = d_scores_;
    int64_t* fiddptr = d_fids_;
    int64_t  nscores = 0;
    /// libids size == 0
    for(int i = 0; i < size; i++){
        int id = libids.at(i);
        FV_DBlocks* bptr = Libsptr(id);
        if(!bptr){
            LOG(WARNING) << "[No. " << gpuid_ << "] GPU has no lib [" << id << "]";
            continue;
        }
        int nblock = bptr->NumLibs();
        for(int i = 0; i < nblock; i++){
            FV_DBlock* blkptr = bptr->data_.at(i);

            /// do verify
            GPUScores(d_matrixswp_, blkptr->matrix_, scoredptr, FV_BLOCKSZ, FV_FEATURESZ);

            /// cpy fid to fids
            __TRYCUDA__(cudaMemcpy(fiddptr, blkptr->fid_, sizeof(int64_t) * FV_BLOCKSZ, cudaMemcpyDeviceToDevice));

            /// update
            nscores   += FV_BLOCKSZ;
            fiddptr   += FV_BLOCKSZ;
            scoredptr += FV_BLOCKSZ;
        }
    }

    /// do thrust sort
    if(nscores > 0){
        /// LOG(INFO) << "No." << gpuid_ << "will do thrust::sort_by_key with num nscores = " << nscores;
        thrust::device_ptr<float>   keys = thrust::device_pointer_cast(d_scores_);
        thrust::device_ptr<int64_t> vals = thrust::device_pointer_cast(d_fids_);
        thrust::sort_by_key(keys, keys + nscores, vals, thrust::greater<float>());

        int count = MIN(nscores, topnum);
        __TRYCUDA__(cudaMemcpy(oscore, d_scores_, sizeof(float)   * count, cudaMemcpyDeviceToHost));
        __TRYCUDA__(cudaMemcpy(ofids,  d_fids_,   sizeof(int64_t) * count, cudaMemcpyDeviceToHost));

        /// LOG(INFO) << "No." << gpuid_ << "get max scores = " << oscore[0] << " min scores = " << oscore[count - 1];
    }

    return true;
}
bool FVDataBase::GPUScores(float* ifeature, float* matrix, float* oscore, int64_t M, int64_t N){
    /// do cublas
    float alpha = 1.0;
    float beta = 0.0;

    cudaEvent_t start;
    cudaEventCreate(&start);
    if(cublasSgemv(  cublas_handle[gpuid_]
                   , CUBLAS_OP_N
                   , M, N
                   , &alpha
                   , matrix
                   , M, ifeature
                   , 1, &beta
                   , oscore, 1)
            != CUBLAS_STATUS_SUCCESS)
        LOG(FATAL) << "cublas sgemv run failed?";

    cudaEventRecord(start);
    cudaEventSynchronize(start);

    /// do score map
    CHECK(fvGPUKernel::ScoreMap(oscore, M));

    return true;
}
bool FVDataBase::ReallocBuf(){
    int nblock = NumBlock();

    if(h_scores_)free(h_scores_); h_scores_ = NULL;
    if(h_fids_)free(h_fids_); h_fids_ = NULL;
    if(d_scores_)__TRYCUDA__(cudaFree(d_scores_)); d_scores_ = NULL;
    if(d_fids_)__TRYCUDA__(cudaFree(d_fids_)); d_fids_ = NULL;

    int64_t size = nblock; size *= FV_BLOCKSZ;

    /// alloc host memory
    if((h_scores_ = (float*)malloc(sizeof(float) * size)) == NULL)
        LOG(FATAL) << "No enough host memory for work!";

    if((h_fids_ = (int64_t*)malloc(sizeof(int64_t) * size)) == NULL)
        LOG(FATAL) << "No enough host memory for work!";

    /// alloc device memory
    __TRYCUDA__(cudaMalloc((void**)&d_scores_, sizeof(float) * size));
    __TRYCUDA__(cudaMalloc((void**)&d_fids_, sizeof(int64_t) * size));

    return true;
}
void FVDataBase::InitScoreMap(){
    float k[4];
    float b[4];

    for(int i = 0; i < 4; i++){
        k[i] = (y[i] - y[i + 1]) / (x[i] - x[i + 1]);
        b[i] = y[i] - k[i] * x[i];
    }

    /// copy k/b/x to gpu
    __TRYCUDA__(cudaMemcpyToSymbol(g_x, x, sizeof(float) * 5));
    __TRYCUDA__(cudaMemcpyToSymbol(g_k, k, sizeof(float) * 4));
    __TRYCUDA__(cudaMemcpyToSymbol(g_b, b, sizeof(float) * 4));
}
