project("SmoothParticleNets")
cmake_minimum_required(VERSION 2.8)

# build release if not explicitly told to build debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_VERBOSE_MAKEFILE on)

set(layer_src
	${PROJECT_SOURCE_DIR}/src/cuda_layer_funcs.c
	${PROJECT_SOURCE_DIR}/src/fgrid.c
	${PROJECT_SOURCE_DIR}/src/gpu_kernels.cu
	${PROJECT_SOURCE_DIR}/src/gpu_kernels.h
	${PROJECT_SOURCE_DIR}/src/spn.h
)

add_custom_command(
    COMMAND ${PYTHON_EXECUTABLE} build.py
    DEPENDS build.py ${layer_src}
    OUTPUT ${PROJECT_SOURCE_DIR}/python/SmoothParticleNets/_ext/__ext.so
    COMMENT "Generating PyTorch code."
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_custom_target(run ALL DEPENDS ${PROJECT_SOURCE_DIR}/python/SmoothParticleNets/_ext/__ext.so)
