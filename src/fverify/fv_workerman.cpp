#include "fv_workerman.h"

using namespace FEATURE_VERIFYNS;
FV_WorkerMan::FV_WorkerMan(int64_t gpumemsz){
    CHECK_GT(gpumemsz, 0);
	int gpuCount = 0;
    int64_t size = gpumemsz * 1024 * 1024;

	cudaGetDeviceCount(&gpuCount);
	for (int i = 0; i < gpuCount; ++i) {
		devices_.push_back(i);
        gpumemsz_.push_back(size);
	}

	InitWorkers();
}
FV_WorkerMan::FV_WorkerMan(std::vector<int>devices, int64_t gpumemsz){
    CHECK_GT(gpumemsz, 0);
    CHECK_GT(devices.size(), 0);

    int64_t size = gpumemsz * 1024 * 1024;

    for(int i = 0; i < devices.size(); i++){
        devices_.push_back(devices.at(i));
        gpumemsz_.push_back(size);
    }

	InitWorkers();
}
FV_WorkerMan::FV_WorkerMan(std::vector<int> devices, std::vector<int64_t> gpumemsz){
    if(devices.size() != gpumemsz.size())
        LOG(FATAL) << "Number of devices ["
                   << devices.size()
                   << "] should equal number of gpumemsz ["
                   << gpumemsz.size() << "]!";

    for(int i = 0; i < devices.size(); i++){
        devices_.push_back(devices.at(i));
        gpumemsz_.push_back(gpumemsz.at(i) * 1024 * 1024);
    }

    InitWorkers();
}

void FV_WorkerMan::InitWorkers() {
    for (int i = 0; i < devices_.size(); ++i) {
        CHECK_GT(gpumemsz_.at(i), 0);
        if(i != 0)CHECK_NE(devices_.at(0), devices_.at(i));

        FV_Worker *worker = new FV_Worker(devices_.at(i), gpumemsz_.at(i));
        workers_.push_back(worker);
    }
    for(int i = 0; i < workers_.size(); i++){
        workers_.at(i)->WaitJobFinish();
        LOG(INFO) << "No." << i + 1 << " worker with gpu[" << devices_.at(i) << "] ready!";
    }

    /// check memory size
    int64_t ret;
    int64_t total = 0;
    for(int i = 0; i < workers_.size(); i++){
        if(NumFeatureCanbeAdded(i, 0, &ret) <= 0)
            LOG(WARNING) << "No." << i + 1
                         << " worker with gpuid ["
                         << devices_.at(i)
                         << "] too small gpu memory! ["
                         << ret << "]";
        else
            LOG(INFO) << "No." << i + 1
                      << " worker with gpuid ["
                      << devices_.at(i) << "] can add ["
                      << ret << "] features!";

        total += ret;
    }
    if(total <= 0)
        LOG(WARNING) << "Too small gpu memory! Cant add features! [" << total << "]";
    else
        LOG(INFO) << "Support max [" << total << "] features!";
}

FV_WorkerMan::~FV_WorkerMan(){
    for(int i = 0; i < workers_.size(); i++)
    {
        FV_Worker *worker = workers_.at(i);
        if(worker)delete worker;
    }

    workers_.clear();
}
int64_t FV_WorkerMan::NumFeatureCanbeAdded(int workid, int libid, int64_t* ret){
    if(workid < 0 || workid >= workers_.size()){
        LOG(ERROR) << "workid value [" << workid << "] should be [0 - " << workers_.size() - 1 << "]!";
        ret[0] = 0;
        return 0;
    }

    workers_[workid]->NumFeatureCanbeAdded(libid, ret);
    workers_[workid]->WaitJobFinish();

    return ret[0];
}
int64_t FV_WorkerMan::NumFeatures(int workid, int libid, int64_t* ret){
    if(workid < 0 || workid >= workers_.size()){
        LOG(ERROR) << "workid value [" << workid << "] should be [0 - " << workers_.size() - 1 << "]!";
        ret[0] = 0;
        return 0;
    }

    workers_[workid]->NumFeatures(libid, ret);
    workers_[workid]->WaitJobFinish();

    return ret[0];
}
int64_t FV_WorkerMan::NumFeatureCanbeAdded(int libid, int64_t* ret) {
    *ret = 0;

    int64_t* workers_ret = new int64_t[workers_.size()];
    for (int i = 0; i < workers_.size(); ++i) {
        workers_[i]->NumFeatureCanbeAdded(libid, workers_ret+i);
    }
	for (int i = 0; i < workers_.size(); ++i) {
        workers_[i]->WaitJobFinish();
	}
	for (int i = 0; i < workers_.size(); ++i) {
		*ret += workers_ret[i];
    }

    return ret[0];
}
int64_t FV_WorkerMan::NumFeatures(int libid, int64_t* ret){
    *ret = 0;

    int64_t* workers_ret = new int64_t[workers_.size()];
    for (int i = 0; i < workers_.size(); ++i) {
        workers_[i]->NumFeatures(libid, workers_ret+i);
    }
    for (int i = 0; i < workers_.size(); ++i) {
        workers_[i]->WaitJobFinish();
    }
    for (int i = 0; i < workers_.size(); ++i) {
        *ret += workers_ret[i];
    }

    return ret[0];
}

int FV_WorkerMan::getAvailiableDevice(int libid) {
	int availiable_device = -1;
	for (int i = 0; i < workers_.size(); ++i) {
		int64_t can_add_num = 0;
		int err = workers_[device_id_]->NumFeatureWanted(libid, &can_add_num);
    	workers_[i]->WaitJobFinish();
		if (can_add_num == FV_BLOCKSZ) {
			availiable_device = device_id_;
			device_id_ = (device_id_+1) % devices_.size();
			break;
		}
		device_id_ = (device_id_+1) % devices_.size();
	}
	return availiable_device;
}

int64_t FV_WorkerMan::AddFeature(int libid, int64_t nfeature, int64_t* fid, float* feature) {
    CHECK_GE(nfeature, 0);

	auto find_lib = [&](int libid)->int {
		for (int i = 0; i < libs_.size(); ++i) {
			if (libs_[i] == libid) return i;
		}
		return -1;
	};
	
    int lib_index = find_lib(libid);
    if (-1 == lib_index) {
        libs_.push_back(libid);
        lib_added_device_id_.push_back(device_id_);
        device_id_ = (device_id_+1) % devices_.size();
        lib_index = libs_.size() - 1;
        std::vector<bool> tmp;
        for (int i = 0; i < workers_.size(); ++i) {
            tmp.push_back(false);
        }
    }
	
    int added_device_id = lib_added_device_id_[lib_index];

    int64_t addTotal = 0;
    int64_t nleft = nfeature;
    while( nleft > 0) {
        int64_t can_add_num = 0;
        int err = workers_[added_device_id]->NumFeatureWanted(libid, &can_add_num);
    	workers_[added_device_id]->WaitJobFinish();
        if (can_add_num == 0) {
            bool is_full = true;
            for(int i = 0; i < workers_.size(); ++i) {
                //int64_t ret = 0;
                workers_[device_id_]->NumFeatureCanbeAdded(libid, &can_add_num);
                workers_[device_id_]->WaitJobFinish();
                if (can_add_num > 0) { is_full = false; break; }
                device_id_ = (device_id_+1) % devices_.size();
            }

            if (is_full) {
                std::cout << "is full" << std::endl;
                break;
            }

            added_device_id = device_id_;
            device_id_ = (device_id_+1)%devices_.size();
            can_add_num = FV_BLOCKSZ;
        }
	
        int64_t addnum = can_add_num > nleft ? nleft : can_add_num;
        workers_[added_device_id]->AddFeature(libid, addnum, fid+addTotal, feature+addTotal*FV_FEATURESZ);
        workers_[added_device_id]->WaitJobFinish();
        //std::cout << libid << " " << added_device_id << " " << addnum << std::endl;
        nleft-= addnum;
        addTotal += addnum;
    }
    
    lib_added_device_id_[lib_index] = added_device_id;
    return addTotal;
}

static void topN(int N, int topnum, float* oscore, int64_t* ofids, const float* workers_oscore, const int64_t* workers_ofids) {
	std::vector<int> idx;
	for (int i = 0; i < N; ++i) {
		idx.push_back(0);
	}
	for (int i = 0; i < topnum; ++i) {
		float ret = 0;
		int ofid = 0;
		int id = 0;
		for (int j = 0; j < N; ++j) {
			if(ret < (workers_oscore+j*topnum)[idx[j]]) {	
				ret = (workers_oscore+j*topnum)[idx[j]];
				ofid = (workers_ofids+j*topnum)[idx[j]];
				id = j;
			}
		}
		idx[id]++;
		oscore[i] = ret;
		ofids[i] = ofid;
	}
}

int FV_WorkerMan::FeatureVerify(float* ifeature, std::vector<int> libids, float* oscore, int64_t* ofids, int topnum) {
	float* workers_oscore = new float[workers_.size()*topnum];
	int64_t* workers_ofids = new int64_t[workers_.size()*topnum];
	for (int i = 0; i < workers_.size(); ++i) {
        workers_[i]->FeatureVerify(ifeature, libids, workers_oscore+i*topnum, workers_ofids+i*topnum, topnum);
	}
	for (int i = 0; i < workers_.size(); ++i) {
        workers_[i]->WaitJobFinish();
	}
	topN(workers_.size(), topnum, oscore, ofids, workers_oscore, workers_ofids);
	delete[] workers_oscore;
	delete[] workers_ofids;
}

int FV_WorkerMan::FeatureVerify(float* ifeature, int libid, float* oscore, int64_t* ofids, int topnum) {
	float* workers_oscore = new float[workers_.size()*topnum];
	int64_t* workers_ofids = new int64_t[workers_.size()*topnum];
	for (int i = 0 ; i < workers_.size(); ++i) {
        workers_[i]->FeatureVerify(ifeature, libid, workers_oscore+i*topnum, workers_ofids+i*topnum, topnum);
	}
	for (int i = 0; i < workers_.size(); ++i) {
        workers_[i]->WaitJobFinish();
	}
	topN(workers_.size(), topnum, oscore, ofids, workers_oscore, workers_ofids);
	delete[] workers_oscore;
	delete[] workers_ofids;
}

int FV_WorkerMan::FeatureVerify(float* ifeature, float* oscore, int64_t* ofids, int topnum) {
	float* workers_oscore = new float[workers_.size()*topnum];
	int64_t* workers_ofids = new int64_t[workers_.size()*topnum];
	for (int i = 0 ; i < workers_.size(); ++i) {
        workers_[i]->FeatureVerify(ifeature, workers_oscore+i*topnum, workers_ofids+i*topnum, topnum);
	}
	for (int i = 0; i < workers_.size(); ++i) {
        workers_[i]->WaitJobFinish();
	}
	topN(workers_.size(), topnum, oscore, ofids, workers_oscore, workers_ofids);
	delete[] workers_oscore;
	delete[] workers_ofids;
}
