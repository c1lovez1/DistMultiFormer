#include <blas.cuh>
#include <utils.cuh>
#include <storage.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cstdlib>


void test_sum_performance(const std::vector<int>& shape, int dim) {
    // 计算总元素数
    int total_elements = 1;
    for (int s : shape) {
        total_elements *= s;
    }
    
    // 创建输入数据
    Storage* input = new Storage(shape);
    thrust::device_vector<float>& input_data = input->get_data();
    
    // 初始化随机数据
    thrust::host_vector<float> host_data(total_elements);
    for (int i = 0; i < total_elements; i++) {
        host_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    input_data = host_data;
    
    // 计算输出形状
    std::vector<int> output_shape = shape;
    output_shape.erase(output_shape.begin() + dim);
    Storage* output = new Storage(output_shape);
    
    // 预热GPU
    operator_sum(input, dim, output);
    cudaDeviceSynchronize();
    
    // 性能测试
    const int num_iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        operator_sum(input, dim, output);
    }
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    float avg_time = duration.count() / (float)num_iterations;
    float throughput = (total_elements * sizeof(float)) / (avg_time * 1000.0f); // GB/s
    
    std::cout << "形状: [";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "], 维度: " << dim << std::endl;
    std::cout << "平均执行时间: " << avg_time << " 微秒" << std::endl;
    std::cout << "吞吐量: " << throughput << " GB/s" << std::endl;
    std::cout << "-------------------" << std::endl;
    
    delete input;
    delete output;
}

int main() {
    // 测试不同大小和维度的情况
    std::vector<std::vector<int>> test_shapes = {
        {32, 32, 32},        // 小型3D张量
        {128, 128, 128},     // 中型3D张量
        {256, 256, 256},     // 大型3D张量
        {16, 16, 16, 16},    // 4D张量
        {32, 32, 32, 32},    // 较大4D张量
        {2, 64, 512, 512}    // 典型的批处理图像大小
    };
    
    for (const auto& shape : test_shapes) {
        for (int dim = 0; dim < shape.size(); dim++) {
            test_sum_performance(shape, dim);
        }
    }
    
    return 0;
}