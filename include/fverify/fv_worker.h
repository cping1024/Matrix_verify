#ifndef _FV_WORKER_H_
#define _FV_WORKER_H_

/*******************************************************************************
 *
 * Copyright © 2016 SenseNets All rights reserved.
 * File name: fv_worker.h
 * Touch time: Wed 11 May 2016 04:49:23 PM CST
 * Author: Yuanpeng Zhang <zhangyuanpeng@sensenets.com>
 * Description:
 * TODO:
 *
*******************************************************************************/

#include "fv_define.h"
#include "glog/logging.h"
#include "fv_database.h"
#include "boost/shared_ptr.hpp"

#include "fv_bqueue.h"
#include "fv_jobtype.h"

#include <cuda_runtime.h>

namespace FEATURE_VERIFYNS {

class FV_Worker{
public:
    FV_Worker(int gpuid, int64_t gpumemsz);
    ~FV_Worker();

    bool Init();
    bool IsOK() {return is_ok_;}
    void StartWorker();
    int  WaitJobFinish();
    bool ExecJob(boost::shared_ptr<BaseJob> job, bool block = false);

    /// ==============================================================
    /// DB action
    /// define DB Query Action
    int  NumFeatureCanbeAdded(int libid, int64_t* ret);
    int  NumFeatureWanted(int libid, int64_t* ret);
    int  NumFeatures(int libid, int64_t* ret);
    int  NumBlocks(int libid, int64_t* ret);
    int  CommitDBQueryCmd(int libid, int code, int64_t* ret);

    /// add DB feature Action
    int  AddFeature(int libid, int64_t nfeature, int64_t* fid, float* feature);

    /// ==============================================================
    /// feature verify action
    int  FeatureVerify(float* ifeature, std::vector<int > libids, float* oscore, int64_t* ofids, int topnum = 100);
    int  FeatureVerify(float* ifeature, int libid, float* oscore, int64_t* ofids, int topnum = 100);
    int  FeatureVerify(float* ifeature, float* oscore, int64_t* ofids, int topnum = 100);

private:

    /// main thread cooperator func
    bool CommitJob(boost::shared_ptr<BaseJob> job);
    bool CommitSysCmd(int code, bool block = false);

    /// =================================
    /// worker thread func
    int  ControlFunc(boost::shared_ptr<BaseJob> job);
    int  InitWorker();
    int  QuitWorker();

    /// define db job action
    int  DoDBJob(boost::shared_ptr<BaseJob> job);
    int  DoDBAddFeatureJob(boost::shared_ptr<DBJob> job);

    /// define fv job action
    int  DoFVJob(boost::shared_ptr<BaseJob> job);

    bool    job_action_;
    bool    is_ok_;
    int     gpuid_;
    int64_t gpu_mem_size_;

    pthread_t   worker_;
    FVDataBase* database_;

    FV_BQueue<boost::shared_ptr<BaseJob> > job_queue_;
    FV_Msg* msg_man_;
}; /// end class FV_Worker

namespace FV_WorkerNS{
    void *ConsumerStarter(void * context);
} /// end namespace FV_WorkerNS

} /// end namespace FEATURE_VERIFYNS




#endif
