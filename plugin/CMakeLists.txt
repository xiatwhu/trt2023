cmake_minimum_required(VERSION 3.6)

project(plugin)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3
            -gencode;arch=compute_80,code=sm_80
            -gencode;arch=compute_86,code=sm_86
            -gencode;arch=compute_86,code=compute_86
            --disable-warnings -use_fast_math -lineinfo -std=c++11 --expt-relaxed-constexpr)

include_directories("${PROJECT_SOURCE_DIR}")

# tensorrt
include_directories("/home/player/TensorRT-8.6.1.6/include")
link_directories("/home/player/TensorRT-8.6.1.6/lib")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O2 -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -Wno-deprecated-declarations")

file(GLOB SOURCE_PLUGINS
	"common/*.cpp"
	"groupNormPlugin/*.cu"
	"groupNormPlugin/*.cpp")
cuda_add_library(myplugins SHARED ${SOURCE_PLUGINS})
target_link_libraries(myplugins nvinfer)
