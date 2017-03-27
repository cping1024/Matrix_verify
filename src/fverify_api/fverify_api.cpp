#include "fverify_api.h"
#include "fv_workerman.h"

using namespace FEATURE_VERIFYNS;

typedef struct _FEATURE_VERIFY_T {
	FV_WorkerMan * worker;
} FEATURE_VERIFY_T, *PFEATURE_VERIFY_T;

FV_SDK_API
fv_handle_t
fv_create_handle(
		fv_config_t config
) {
	std::vector<int> devices;
	std::vector<int64_t> gpumemsz;
	for (int i = 0; i < config.N; ++i) {
		devices.push_back(config.devices[i]);
		gpumemsz.push_back(config.memsz[i]);
	}
	PFEATURE_VERIFY_T inst = new FEATURE_VERIFY_T();
	if (!inst) {
		return NULL;
	}
	inst->worker = new FV_WorkerMan(devices, gpumemsz);
	if (!inst->worker) {
		return NULL;
	}
	return inst;
}

FV_SDK_API
void fv_destroy_handle(
	fv_handle_t handle
) {
	if (!handle) {
		return;
	}
	
	PFEATURE_VERIFY_T inst = static_cast<PFEATURE_VERIFY_T>(handle);
	
	if (inst->worker) {
	    delete inst->worker;
	}
	
	delete inst;
}

FV_SDK_API
int64_t fv_get_available_capacity(
	fv_handle_t handle,
	int libid
) {
	if (!handle) {
		return 0;
	}
	PFEATURE_VERIFY_T inst = static_cast<PFEATURE_VERIFY_T>(handle);
	int64_t ret = 0;
	return inst->worker->NumFeatureCanbeAdded(libid, &ret);
}

FV_SDK_API
int64_t fv_add_feature(
	fv_handle_t handle,
	int lib,
	int64_t num,
	int64_t* fid,
	float* features
) {
	if (!handle) {
		return 0;
	}
	PFEATURE_VERIFY_T inst = static_cast<PFEATURE_VERIFY_T>(handle);
	return inst->worker->AddFeature(lib, num, fid, features);
}


FV_SDK_API
void fv_verify(
	fv_handle_t handle,
	float* feature,
	int libid,
	float* oscore,
	int64_t* fid,
	int topnum
) {
	if (!handle) {
		return;
	}
	PFEATURE_VERIFY_T inst = static_cast<PFEATURE_VERIFY_T>(handle);
	inst->worker->FeatureVerify(feature, libid, oscore, fid, topnum);
}


FV_SDK_API
void fv_verify(
	fv_handle_t handle,
	float* feature,
	int nlibs,
	int* libids,
	float* oscore,
	int64_t* fid,
	int topnum
) {
	if (!handle) {
		return;
	}
	PFEATURE_VERIFY_T inst = static_cast<PFEATURE_VERIFY_T>(handle);
	std::vector<int> libs;
	for (int i = 0; i < nlibs; ++i) {
		libs.push_back(libids[i]);
	}
	inst->worker->FeatureVerify(feature, libs, oscore, fid, topnum);
}


FV_SDK_API
void fv_verify(
	fv_handle_t handle,
	float* feature,
	float* oscore,
	int64_t* fid,
	int topnum
) {
	if (!handle) {
		return;
	}
	PFEATURE_VERIFY_T inst = static_cast<PFEATURE_VERIFY_T>(handle);
	inst->worker->FeatureVerify(feature, oscore, fid, topnum);
}
