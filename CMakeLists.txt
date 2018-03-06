project("SmoothParticleNets")
cmake_minimum_required(VERSION 2.8)

# build release if not explicitly told to build debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_VERBOSE_MAKEFILE on)

find_package(CUDA)

set(layer_src
	${PROJECT_SOURCE_DIR}/src/cuda_layer_funcs.c
	${PROJECT_SOURCE_DIR}/src/cpu_layer_funcs.c
	${PROJECT_SOURCE_DIR}/src/gpu_kernels.cu
	${PROJECT_SOURCE_DIR}/src/gpu_kernels.h
	${PROJECT_SOURCE_DIR}/src/spn.h
	${PROJECT_SOURCE_DIR}/src/cuda_spn.h
	${PROJECT_SOURCE_DIR}/src/non_cuda_spn.h
    ${PROJECT_SOURCE_DIR}/src/constants.h
    ${PROJECT_SOURCE_DIR}/src/common_funcs.h
    ${PROJECT_SOURCE_DIR}/python/SmoothParticleNets/kernels.py
)

set(args 
)
if(CUDA_FOUND)
    set(args
        ${args}
        --with_cuda
    )
endif()

if(CMAKE_BUILD_TYPE EQUAL "DEBUG")
    set(args
        ${args}
        --debug
    )
endif()

add_custom_command(
    COMMAND ${PYTHON_EXECUTABLE} build.py ${args}
    DEPENDS build.py ${layer_src}
    OUTPUT ${PROJECT_SOURCE_DIR}/python/SmoothParticleNets/_ext/__ext.so
    COMMENT "Generating PyTorch code."
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_custom_target(run ALL DEPENDS ${PROJECT_SOURCE_DIR}/python/SmoothParticleNets/_ext/__ext.so)
