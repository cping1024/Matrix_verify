#include "fverify_api.h"
#include <iostream>
#include <cassert>
#include <glog/logging.h>

#include <cuda_runtime.h>
#define FV_FEATURESZ 133

void init_array(float *a, int64_t N){
    assert(a != nullptr);

    for(int64_t i = 0; i < N; i++)
        a[i] = 0.1 * rand() / RAND_MAX;
}

int test_fvapi(){
	fv_config_t config;
	config.N = 1;
	assert(config.N <= MAX_DEVICE_NUM);
	for (int i = 0; i < config.N; i++) {
		config.devices[i] = i;
		config.memsz[i] = 10*1024;
	}

	fv_handle_t handle = fv_create_handle(config);
	std::cout << "create api handle !" << std::endl;
	
    int64_t ret = 0;
    int64_t num_feature = 600 * 10000;
    float* features = new float[num_feature * FV_FEATURESZ];
    int64_t* fid = new int64_t[num_feature];

    for(int64_t i = 0; i < num_feature; i++)fid[i] = i + 1;

    init_array(features, num_feature * FV_FEATURESZ);
	ret = fv_add_feature(handle, 0, num_feature, fid, features);
	std::cout << "lib 0 add " << ret << " success." << std::endl;
  	/*	
	ret = fv_add_feature(handle, 1, num_feature, fid, features);
	std::cout << "lib 1 add " << ret << " success." << std::endl;
	
	ret = fv_add_feature(handle, 2, num_feature, fid, features);
	std::cout << "lib 2 add " << ret << " success." << std::endl;
	
	ret = fv_add_feature(handle, 3, num_feature, fid, features);
	std::cout << "lib 3 add " << ret << " success." << std::endl;
	
	std::cout << "---------------------------------------------" << std::endl;
	ret = fv_get_available_capacity(handle, 0);
	std::cout << "lib 0 can be added " << ret << std::endl;
	
	ret = fv_get_available_capacity(handle, 1);
	std::cout << "lib 1 can be added " << ret << std::endl;
	
	ret = fv_get_available_capacity(handle, 2);
	std::cout << "lib 2 can be added " << ret << std::endl;
	
	ret = fv_get_available_capacity(handle, 3);
	std::cout << "lib 3 can be added " << ret << std::endl;
	*/
	float* feature = new float[FV_FEATURESZ];
	init_array(feature, FV_FEATURESZ);

	float* oscore = new float[100];
	int64_t* ofid = new int64_t[100];

	fv_verify(handle, feature, oscore, ofid);
	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 10; ++j) {
			std::cout << oscore[i*10+j] << " ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
	for (int i = 0 ; i < 10; ++i) {
		for (int j = 0; j < 10; ++j) {
			std::cout << ofid[i*10+j] << " ";
		}
		std::cout << std::endl;
	}

	fv_destroy_handle(handle);
	delete[] features;
	delete[] fid;
	delete[] feature;
	delete[] oscore;
	delete[] ofid;
	return 0;
}


int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    google::LogToStderr();

    return test_fvapi();
}
