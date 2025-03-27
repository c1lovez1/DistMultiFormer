cmake_minimum_required(VERSION 3.10)
project(operator_performance CUDA CXX)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 设置编译选项
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA编译选项
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)

# 查找CUDA包
find_package(CUDA REQUIRED)

# 添加include目录
include_directories(
    ${CMAKE_SOURCE_DIR}/..
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 添加可执行文件
add_executable(operator_sum_perf operator_sum.cu)

# 链接CUDA库
target_link_libraries(operator_sum_perf
    ${CUDA_LIBRARIES}
)

# 设置CUDA编译属性
set_target_properties(operator_sum_perf PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)