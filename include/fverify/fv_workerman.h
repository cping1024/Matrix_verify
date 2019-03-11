#ifndef _FV_WORKER_MAN_H_
#define _FV_WORKER_MAN_H_

#include "fv_define.h"
#include "fv_worker.h"

namespace FEATURE_VERIFYNS {

class FV_WorkerMan{
public:
    /// default: use all gpu
    /// each gpu memory used will be min(gpuFreeMemory, gpumemsz) in MB
    FV_WorkerMan(int64_t gpumemsz = 10 * 1024);

    /// user defined gpu id, [0, 1, ... n] where n <= numGPU
    /// each gpu memory used will be min(gpuFreeMemory, gpumemsz) in MB
    FV_WorkerMan(std::vector<int> devices, int64_t gpumemsz = 10 * 1024);

    /// user specify gpu id and how much gpu memory can be used in MB
    FV_WorkerMan(std::vector<int> devices, std::vector<int64_t> gpumemsz);
    ~FV_WorkerMan();

    int numWorker() {return workers_.size();}
    int64_t NumFeatureCanbeAdded(int workid, int libid, int64_t* ret);
    int64_t NumFeatures(int workid, int libid, int64_t* ret);
    int64_t NumFeatureCanbeAdded(int libid, int64_t* ret);
    int64_t NumFeatures(int libid, int64_t* ret);
    int64_t AddFeature(int libid, int64_t nfeature, int64_t* fid, float* feature);

    int FeatureVerify(float* ifeature, std::vector<int> libids, float* oscore, int64_t* ofids, int topnum = 100);
    int FeatureVerify(float* ifeature, int libid, float* oscore, int64_t* ofids, int topnum = 100);
    int FeatureVerify(float* ifeature, float* oscore, int64_t* ofids, int topnum = 100);

private:
    void InitWorkers();
    int getAvailiableDevice(int libid);

    std::vector<int> devices_;
    std::vector<int> libs_;
    std::vector<int> lib_added_device_id_;
    std::vector<int64_t> gpumemsz_;

    std::vector<FV_Worker* > workers_;
    std::vector<bool> next_device_id_;
    std::vector<std::vector<bool> >libs_devices_full_;
    int device_id_ = 0;
}; /// end class FV_WorkerMan

} /// end namespace FEATURE_VERIFYNS

#endif

