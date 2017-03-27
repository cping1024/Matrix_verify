#ifndef _FEATURE_FVERIFY_H_
#define _FEATURE_FVERIFY_H_

#define FV_SDK_API
#define MAX_DEVICE_NUM 8

#include<cstdint>

typedef void* fv_handle_t;

typedef struct fv_config_t {
	int devices[MAX_DEVICE_NUM];
	int64_t memsz[MAX_DEVICE_NUM];
	int N;
} fv_config_t;

FV_SDK_API
fv_handle_t
fv_create_handle(
	fv_config_t config
);

FV_SDK_API
void fv_destroy_handle(
	fv_handle_t handle
);

FV_SDK_API
int64_t fv_get_available_capacity(
	fv_handle_t handle,
	int libid);

FV_SDK_API
int64_t fv_add_feature(
	fv_handle_t handle,	
	int lib,
	int64_t num,
	int64_t* fid,
	float* features
);

FV_SDK_API
void fv_verify(
	fv_handle_t handle,	
	float* feature,
	int libid,
	float* oscore,
	int64_t* ofid, int topnum = 100
);

FV_SDK_API
void fv_verify(
	fv_handle_t handle,	
	float* feature,
	int nlibs,
	int* libid,
	float* oscore,
	int64_t* ofid, int topnum = 100
);

FV_SDK_API
void fv_verify(
	fv_handle_t handle,	
	float* feature,
	float* oscore,
	int64_t* ofid, int topnum = 100
);

#endif // _FEATURE_FVERIFY_H_
