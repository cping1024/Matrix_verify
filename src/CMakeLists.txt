FILE(GLOB_RECURSE CPP_SRCS *.cpp)
FILE(GLOB_RECURSE CU_SRCS *.cu)
CUDA_COMPILE(CUDA_OBJS ${CU_SRCS})
ADD_LIBRARY(fverify SHARED ${CPP_SRCS} ${CUDA_OBJS})
TARGET_LINK_LIBRARIES(fverify ${FVERIFY_LINKER_LIBS})
SET(FVERIFY_LINKER_LIBS fverify ${FVERIFY_LINKER_LIBS} PARENT_SCOPE)
SEt_TARGET_PROPERTIES(fverify PROPERTIES VERSION ${FVERIFY_VERSION} SVERSION ${FVERIFY_MAJOR_VERSION})
INSTALL(TARGETS fverify DESTINATION lib)


