#ifndef _FV_JOB_TYPE_H_
#define _FV_JOB_TYPE_H_

/*******************************************************************************
 *
 * Copyright © 2016 SenseNets All rights reserved.
 * File name: fv_worker.h
 * Touch time: Fri 13 May 2016 04:49:23 PM CST
 * Author: Yuanpeng Zhang <zhangyuanpeng@sensenets.com>
 * Description:
 * TODO:
 *
*******************************************************************************/

#include "fv_define.h"

namespace FEATURE_VERIFYNS{

/// ====================================
/**
  * Base job 0-99
  * thread manager, do sys cmd
  * ---------------------
  *
  *   0: init
  *  99: exit
 */
class BaseJob {
public:
    BaseJob(int type = -1)
        : job_type_(type){}

    virtual ~BaseJob(){}

    BaseJob(BaseJob* ref){
        if(ref)job_type_ = ref->job_type_;
        else
            job_type_ = -1;
    }

    int   job_type_;
};

/// ====================================
/**
  * database job 100-199
  * database manager
  * ---------------------
  *
  * 110: report current num of blocks in device mem
  * 111: report current num of features
  * 120: report num of feature can be added
  * 121: report num of feature wanted: (for current free block)
  * 130: add features, need size, features_ptr, fid_ptr, libid ...
  *      here libid cant equal -1, must specify libid!
 */
struct DBJobParam{
    DBJobParam()
        : libid_(-1)
        , nfeatures_(0)
        , features_(NULL)
        , fids_(NULL){}

    /// working on which lib, -1 means all lib
    int libid_;

    /// number of feature will added?
    int64_t nfeatures_;

    /// feature ptrs
    float* features_;

    /// feature fid
    int64_t* fids_;

    /// report/output ptr, thread will write report to
    int64_t* report;
};
class DBJob : public BaseJob{
public:
    DBJob(int type)
        : BaseJob(type)
    {}

    virtual ~DBJob(){}

    DBJobParam param_;
};

/// ====================================
/** operator job 200-299
  * ---------------------
  *
  * 200: feature verify
  *
  */
struct FVJobParam{
    FVJobParam()
        : nlibs_(-100)
        , top_num_(100)
        , ifeature_(NULL)
        , oscore_(NULL)
        , ofid_(NULL){}

    /// verify in nlibs, == -1, will verify in all libs
    ///                  == -100, nothing todo  /// to ugly! need edit it later! by yuanpengzhang
    int nlibs_;

    /// top number;
    int top_num_;

    /// specify libid, size = nlibs if nlibs != -1
    std::vector<int > libids_;

    /// feature ptr which will be verfied
    float* ifeature_;

    /// result/output ptr: score & fid, where slave will write to
    float*   oscore_;
    int64_t* ofid_;
};
class FVJob : public BaseJob {
public:
    FVJob(int type)
        : BaseJob(type)
    {}

    virtual ~FVJob(){}

    FVJobParam param_;
};

} /// end namespace FEATURE_VERIFYNS



#endif
