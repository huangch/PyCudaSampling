# MakeLists.txt

cmake_minimum_required(VERSION 3.0)
project(cudaSampling)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
 
# # This comes to hand if we also need to use the NumPy C API
# exec_program(${PYTHON_EXECUTABLE}
#              ARGS "-c \"import numpy; print(numpy.get_include())\""
#              OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
#              RETURN_VALUE NUMPY_NOT_FOUND
#             )
# if(NUMPY_NOT_FOUND)
#     message(FATAL_ERROR "NumPy headers not found")
# endif()

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-options -fPIC)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --cudart=static) 
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_52)

cuda_add_library(cudaSampling STATIC cudaSampling.cu)
# cuda_add_library(cudaSampling SHARED cudaSampling.cu)

# target_link_libraries(cudaSampling)
