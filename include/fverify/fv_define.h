#ifndef _FV_DEFINE_H_
#define _FV_DEFINE_H_

/*******************************************************************************
 *
 * Copyright © 2016 SenseNets All rights reserved.
 * File name: fv_define.h
 * Touch time: Wed 11 May 2016 04:49:23 PM CST
 * Author: Yuanpeng Zhang <zhangyuanpeng@sensenets.com>
 * Description:
 * TODO: nothing
 *
*******************************************************************************/

#include <assert.h>
#include <vector>
#include <iostream>

#include <cublas_v2.h>

namespace FEATURE_VERIFYNS {

#ifndef STFACE_VER_520
#define STFACE_VER_520
#endif

#ifndef FV_BLOCKSZ
#define FV_BLOCKSZ  65535
#endif

#ifndef FV_FEATURESZ
#define FV_FEATURESZ  133
#endif

#ifndef FV_MAXLIBNUM
#define FV_MAXLIBNUM   150
#endif

#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

/// senseTime score map
#ifdef STFACE_VER_472
    float x[5] = { -1, 0.39, 0.44, 0.5, 1 };
    float y[5] = { 0, 0.5, 0.7, 0.9, 1 };
#endif //STFACE_VER_472

#ifdef STFACE_VER_520
    static float x[5] = { -1, 0.43, 0.5, 0.55, 1 };
    static float y[5] = { 0, 0.5, 0.7, 0.9, 1 };

#define X0  -1
#define X1   0.43
#define X2   0.5
#define X3   0.55
#define X4   1

#define K0   0.349650323390961
#define K1   2.857142925262451
#define K2   3.999998807907104
#define K3   0.222222283482552

#define B0   0.349650323390961
#define B1  -0.728571534156799
#define B2  -1.299999475479126
#define B3   0.777777731418610

#endif //STFACE_VER_520

#define __TRYCUDA__(a)  { \
    cudaError_t e = a; \
    if(e != cudaSuccess)\
        LOG(FATAL) << "CUDA run_time error: " << cudaGetErrorString(e);\
    }

#ifndef FV_MAXDEVICENUM
#define FV_MAXDEVICENUM 16
#endif

static cublasHandle_t cublas_handle[FV_MAXDEVICENUM];

}


#endif
