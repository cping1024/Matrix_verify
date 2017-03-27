#include "fv_worker.h"

using namespace  FEATURE_VERIFYNS;
using namespace  FV_WorkerNS;
FV_Worker::FV_Worker(int gpuid, int64_t gpumemsz)
    : job_action_(false)
    , is_ok_(false)
    , gpuid_(gpuid)
    , gpu_mem_size_(gpumemsz)
    , database_(NULL)
    , msg_man_(new FV_Msg())
{
    Init();
}
FV_Worker::~FV_Worker(){
    /// quit worker thread
    CommitSysCmd(99, true);

    /// free memory
    if(msg_man_)delete msg_man_; msg_man_ = NULL;
}
bool FV_Worker::Init(){
    CHECK(!IsOK()) << "Double init? Dont support in this func!";

    /// start worker thread
    if(pthread_create(&worker_, NULL, ConsumerStarter, (void*)this))
        LOG(FATAL) << "Can't start worker thread: " << gpuid_ << std::endl;

    /// commit worker init command
    CommitSysCmd(0);

    return true;
}
bool FV_Worker::ExecJob(boost::shared_ptr<BaseJob> job, bool block){
    WaitJobFinish();
    CommitJob(job);
    if(!block) return true;

    return WaitJobFinish();
}
int  FV_Worker::NumBlocks(int libid, int64_t* ret){
    return CommitDBQueryCmd(libid, 110, ret);
}
int  FV_Worker::NumFeatures(int libid, int64_t* ret){
    return CommitDBQueryCmd(libid, 111, ret);
}
int  FV_Worker::NumFeatureCanbeAdded(int libid, int64_t* ret){
    return CommitDBQueryCmd(libid, 120, ret);
}
int  FV_Worker::NumFeatureWanted(int libid, int64_t* ret){
    return CommitDBQueryCmd(libid, 121, ret);
}
int  FV_Worker::CommitDBQueryCmd(int libid, int code, int64_t* ret){
    boost::shared_ptr<DBJob> jobptr(new DBJob(code));

    jobptr->param_.libid_ = libid;
    jobptr->param_.report = ret;

    ExecJob(jobptr, false);

    return 0;
}
int  FV_Worker::AddFeature(int libid, int64_t nfeature, int64_t* fid, float* feature){
    CHECK_GT(nfeature, 0);

    boost::shared_ptr<DBJob> jobptr(new DBJob(130));

    jobptr->param_.libid_ = libid;
    jobptr->param_.nfeatures_ = nfeature;
    jobptr->param_.features_  = feature;
    jobptr->param_.fids_      = fid;

    ExecJob(jobptr, false);

    return 0;
}
int  FV_Worker::FeatureVerify(float* ifeature, std::vector<int > libids, float* oscore, int64_t* ofids, int topnum){    
    boost::shared_ptr<FVJob> jobptr(new FVJob(200));

    jobptr->param_.nlibs_ = libids.size();
    jobptr->param_.top_num_ = topnum;
    jobptr->param_.libids_.insert(jobptr->param_.libids_.end(), libids.begin(), libids.end());
    jobptr->param_.ifeature_ = ifeature;
    jobptr->param_.oscore_   = oscore;
    jobptr->param_.ofid_     = ofids;

    ExecJob(jobptr, false);

    return 0;
}
int  FV_Worker::FeatureVerify(float* ifeature, int libid, float* oscore, int64_t* ofids, int topnum){
    boost::shared_ptr<FVJob> jobptr(new FVJob(200));

    jobptr->param_.nlibs_ = 1;
    jobptr->param_.top_num_ = topnum;
    jobptr->param_.libids_.push_back(libid);
    jobptr->param_.ifeature_ = ifeature;
    jobptr->param_.oscore_   = oscore;
    jobptr->param_.ofid_     = ofids;

    ExecJob(jobptr, false);

    return 0;
}
int  FV_Worker::FeatureVerify(float* ifeature, float* oscore, int64_t* ofids, int topnum){
    boost::shared_ptr<FVJob> jobptr(new FVJob(200));

    jobptr->param_.nlibs_ = -1;
    jobptr->param_.top_num_ = topnum;
    jobptr->param_.ifeature_ = ifeature;
    jobptr->param_.oscore_   = oscore;
    jobptr->param_.ofid_     = ofids;

    ExecJob(jobptr, false);

    return 0;
}
/// ===========================================
/// main thread func
int  FV_Worker::WaitJobFinish(){
    if(!job_action_)return true;

    std::unique_lock <std::mutex> lck(msg_man_->mutex_);
    while(msg_man_->status_ == 1)
        msg_man_->condition_.wait(lck);

    job_action_ = false;

    return msg_man_->report_;
}
bool FV_Worker::CommitJob(boost::shared_ptr<BaseJob> job){
    CHECK_EQ(job_action_, false);
    job_action_ = true;

    std::unique_lock <std::mutex> lck(msg_man_->mutex_);
    msg_man_->status_ = 1;

    job_queue_.push(job);

    return true;
}
bool FV_Worker::CommitSysCmd(int code, bool block){
    /// create job
    boost::shared_ptr<BaseJob> jobptr(new BaseJob(code));

    ExecJob(jobptr, block);

    return true;
}

/// ===========================================
/// worker thread func
void FV_Worker::StartWorker(){
    while(true){
        /// get job from queue
        boost::shared_ptr<BaseJob> job = job_queue_.pop();

        if(job->job_type_ == 99){ /// do quit act
            /// free buff
            int ret = QuitWorker();
            msg_man_->ChangeStatus(0, ret);
            return;
        }else if(job->job_type_ == 0){ /// do init act
            int ret = InitWorker();
            msg_man_->ChangeStatus(0, ret);
            continue;
        }else{
            int ret = ControlFunc(job);
            msg_man_->ChangeStatus(0, ret);
            continue;
        }
    }
}
int  FV_Worker::ControlFunc(boost::shared_ptr<BaseJob> job){
    if(job == NULL)return 0;

    int ret = 0;
    switch(job->job_type_ / 100){
    case 1:
        ret = DoDBJob(job);
        break;
    case 2:
        ret = DoFVJob(job);
        break;
    default:
        break;
    }

    return ret;
}
int  FV_Worker::QuitWorker(){
    if(database_)delete database_; database_ = NULL;

    return 1;
}
int  FV_Worker::InitWorker(){    
    if(IsOK())return 1;

#define CHECKERR(a){ \
    err = a;\
    if(err != cudaSuccess){\
        LOG(ERROR) << "CUDA run time error: " << cudaGetErrorString(err);\
        return 0;\
    }\
}
    LOG(INFO) << "No." << gpuid_ << " setup device ...";
    /// set device
    cudaError err;

    int count;
    CHECKERR(cudaGetDeviceCount(&count));

    if(gpuid_ >= count || gpuid_ < 0){
        LOG(ERROR) << "Bad parameter gpuid: " << gpuid_ << ". Expected [0 - " << count << "].";
        return 0;
    }

    CHECKERR(cudaSetDevice( gpuid_ ));

    /// adjust gpu_mem_size
    size_t freesize, totalsize;
    CHECKERR(cudaMemGetInfo(&freesize, &totalsize));
    if(gpu_mem_size_ >= freesize){
        LOG(WARNING) << "No." << gpuid_
                     << " GPU free memory is ["
                     << freesize / 1024 / 1024 << " MB]."
                     << "Has changed gpu mem_size from "
                     << gpu_mem_size_ / 1024 / 1024
                     << " to " << freesize / 1024 / 1024;

        gpu_mem_size_ = freesize;
    }

    /// alloc database_
    LOG(INFO) << "No." << gpuid_ << " alloc memory ...";
    database_ = new FVDataBase(gpuid_, gpu_mem_size_);

#undef CHECKERR    

    is_ok_ = true;
    return 1; /// successful
}
int  FV_Worker::DoDBJob(boost::shared_ptr<BaseJob> job){
    boost::shared_ptr<DBJob> dbjob = boost::dynamic_pointer_cast<DBJob>(job);

    int ret = 1;
    switch(dbjob->job_type_){
    case 110:
        /// report current num of blocks in device mem
        dbjob->param_.report[0] = database_->NumBlock(dbjob->param_.libid_);
        break;
    case 111:
        /// report current num of features
        dbjob->param_.report[0] = database_->NumFeatures(dbjob->param_.libid_);
        break;
    case 120:
        /// report num of feature can be added        
        dbjob->param_.report[0] = database_->NumFeatureCanbeAdded(dbjob->param_.libid_);
        break;
    case 121:
        /// report num of feature wanted        
        dbjob->param_.report[0] = database_->NumFeatureWanted(dbjob->param_.libid_);
        break;
    case 130:
        /// add features
        ret = DoDBAddFeatureJob(dbjob);
        break;
    default:
        LOG(ERROR) << "Unknow command code [" << dbjob->job_type_ << "]";
        ret = 0;
    }

    return ret; /// successful
}
int  FV_Worker::DoDBAddFeatureJob(boost::shared_ptr<DBJob> job){
    CHECK(job != NULL) << "BUG: Empty ptr?";
    CHECK_EQ(job->job_type_, 130) << "BUG: Bad command code!";

    bool ret;
    if(job->param_.nfeatures_ == 1)
        ret = database_->AddFeature(  job->param_.libid_
                                    , job->param_.fids_[0]
                                    , job->param_.features_);
    else
        ret = database_->AddFeature(  job->param_.libid_
                                    , job->param_.nfeatures_
                                    , job->param_.fids_
                                    , job->param_.features_);

    return ret ? 1 : 0;
}
int  FV_Worker::DoFVJob(boost::shared_ptr<BaseJob> job){
    CHECK_EQ(job->job_type_, 200);
    boost::shared_ptr<FVJob> fvjob = boost::dynamic_pointer_cast<FVJob>(job);

    bool ret;
    switch(fvjob->job_type_){
    case 200:
        /// do feature verify job
        if(fvjob->param_.nlibs_ == -1)
            ret = database_->FeatureVerify(  fvjob->param_.ifeature_
                                           , fvjob->param_.oscore_
                                           , fvjob->param_.ofid_
                                           , fvjob->param_.top_num_);
        else
            ret = database_->FeatureVerify(  fvjob->param_.ifeature_
                                           , fvjob->param_.libids_
                                           , fvjob->param_.oscore_
                                           , fvjob->param_.ofid_
                                           , fvjob->param_.top_num_);
        break;
    default:
        LOG(ERROR) << "Unknow command code [" << job->job_type_ << "]";
        ret = false;
    }

    return ret ? 1 : 0; /// successful
}
void *FV_WorkerNS::ConsumerStarter(void* context){
    ((FV_Worker *)context)->StartWorker();
    pthread_exit(0);

    return NULL;
}
