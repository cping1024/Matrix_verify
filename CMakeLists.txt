CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(feature-verify C CXX)

SET(FVERIFY_MAJOR_VERSION 1)
SET(FVERIFY_MINOR_VERSION 0)
SET(FVERIFY_PATCH_VERSION 0)

SET(FVERIFY_VERSION ${FVERIFY_MAJOR_VERSION}.${FVERIFY_MINOR_VERSION}.${FVERIFY_PATCH_VERSION})

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

option(BUILD_EXAMPLES "build examples" ON)
option(BUILD_TESTS "build test cases" OFF)

FIND_PACKAGE( OpenMP REQUIRED)
INCLUDE(cmake/dependencies.cmake)

SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-g -fPIC -std=c++11")
SET(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-g -fPIC")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-g -Xcompiler -fPIC -std=c++11
    -gencode=arch=compute_20,code=\"sm_20\"
    -gencode=arch=compute_20,code=\"sm_21\"
    -gencode=arch=compute_30,code=\"sm_30\"
    -gencode=arch=compute_35,code=\"sm_35\"
    -gencode=arch=compute_50,code=\"sm_50\"
    -gencode=arch=compute_50,code=\"compute_50\"
    ")

SET(FV_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
LIST(APPEND FV_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/fverify)
LIST(APPEND FV_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/fverify_api)

INCLUDE_DIRECTORIES(${FV_INCLUDE_DIR})

ADD_SUBDIRECTORY(src)

SET(FV_LINK fverify_api)

IF(BUILD_EXAMPLES)
    ADD_SUBDIRECTORY(example)
ENDIF()

INSTALL(FILES DESTINATION lib)
INSTALL(DIRECTORY include/fverify_api DESTINATION include/)
