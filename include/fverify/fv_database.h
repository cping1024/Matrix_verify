#ifndef _FV_DATABASE_H_
#define _FV_DATABASE_H_

/*******************************************************************************
 *
 * Copyright © 2016 SenseNets All rights reserved.
 * File name: fv_database.h
 * Touch time: Wed 11 May 2016 04:49:23 PM CST
 * Author: Yuanpeng Zhang <zhangyuanpeng@sensenets.com>
 * Description:
 * TODO: 1 block size feature add!
 *
*******************************************************************************/

#include <cuda_runtime.h>

#include "fv_define.h"
#include "glog/logging.h"

#include <map>

namespace FEATURE_VERIFYNS {

class FV_DBlock {
public:
    FV_DBlock(){
        count_ = 0;

        int64_t size = sizeof(float) * FV_BLOCKSZ * FV_FEATURESZ;

        __TRYCUDA__(cudaMalloc((void**)&fid_, sizeof(int64_t) * FV_BLOCKSZ));
        __TRYCUDA__(cudaMalloc((void**)&matrix_, size));
        __TRYCUDA__(cudaMemset(matrix_, 0, size));
    }
    ~FV_DBlock(){
        if(fid_)__TRYCUDA__(cudaFree(fid_));
        if(matrix_)__TRYCUDA__(cudaFree(matrix_));
    }

    int Count(){return count_;}
    static int64_t Size(){
        int64_t ret = 0;
        ret += sizeof(int64_t) * FV_BLOCKSZ;  /// fid
        ret += sizeof(float)   * FV_BLOCKSZ * FV_FEATURESZ; /// matrix

        return ret;
    }

    int       count_;
    int64_t*  fid_;
    float*    matrix_;
};

class FV_DBlocks {
public:
    FV_DBlocks(int gpuid, int libid, float* h_swp, float* d_swp)
        : gpuid_(gpuid)
        , libid_(libid)
        , h_swp_(h_swp)
        , d_swp_(d_swp)
        , size_(0)
    {
        CHECK(h_swp_ != NULL);
        CHECK(d_swp_ != NULL);
    }

    ~FV_DBlocks(){
        int nlib = data_.size();
        for(int i = 0; i < nlib; i++){
            FV_DBlock* blk = data_.at(i);
            if(blk)delete blk;
        }
        data_.clear();
    }

    bool AddFeature(int64_t nfeature, int64_t* fid, float* features);
    bool AddFeature(int64_t fid, float* feature);
    int64_t Size() {return size_;}

    FV_DBlock* NxtBlockptr();

    int64_t NumFeatures();
    int LibID() {return libid_;}
    int GPUID() {return gpuid_;}
    int NumLibs() {return data_.size();}

    int           gpuid_;
    int           libid_;

    float*        h_swp_;
    float*        d_swp_;
    int64_t       size_;
    std::vector<FV_DBlock*> data_;
};

class FVDataBase {
public:
    FVDataBase(int gpuid, int64_t gpumemsz);
    ~FVDataBase();

    bool AddFeature(int libid, int nfeature, int64_t* fid, float* features);
    bool AddFeature(int libid, int64_t fid, float* feature);
    int64_t NumFeatureCanbeAdded(int libid);
    int64_t NumFeatureWanted(int libid);

    bool FeatureVerify(float* ifeature, int libid, float* oscore, int64_t* ofids, int topnum = 100);
    bool FeatureVerify(float* ifeature, float* oscore, int64_t* ofids, int topnum = 100);
    bool FeatureVerify(float* ifeature, std::vector<int > libids, float* oscore, int64_t* ofids, int topnum = 100);

    int64_t UsedMemSize();
    int NumLibs() {return libs_.size();}

    int64_t NumFeatures(int libid = -1);
    int NumBlock(int libid = -1);

    FV_DBlocks* Libsptr(int libid);

private:
    void InitScoreMap();
    bool GPUScores(float* ifeature, float* matrix, float* oscore, int64_t M, int64_t N);
    bool ExistLib(int libid);
    bool AddLibDef(int libid);
    bool ReallocBuf();
    int  LibIndex(int libid);
    inline int64_t DSwpSize(){
        return sizeof(float) * FV_BLOCKSZ * FV_FEATURESZ * 2;
    }

    int  gpuid_;
    std::vector<FV_DBlocks* > libs_;

    int64_t  gpu_mem_sz_;
    float*   h_matrixswp_;
    float*   d_matrixswp_;

    float*   h_scores_;
    float*   d_scores_;
    int64_t* h_fids_;
    int64_t* d_fids_;
};


} /// end namespace  FEATURE_VERIFYNS

#endif
