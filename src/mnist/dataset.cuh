#pragma once

// 包含必要的头文件
#include <layer.cuh>              // 神经网络层的基类
#include <thrust/host_vector.h>   // Thrust主机向量
#include <thrust/system/cuda/experimental/pinned_allocator.h>  // CUDA固定内存分配器
#include <memory>                 // 智能指针
#include <string>                 // 字符串
#include <utility>                // 包含pair、move等工具函数
#include <vector>                 // 向量容器
#include <iostream>               // 输入输出流

// MNIST数据集加载类,继承自Layer基类
class DataSet : public Layer {
public:
    // 构造函数,传入MNIST数据路径和是否打乱数据的标志
    explicit DataSet(std::string minist_data_path, bool shuffle = false);
    
    // 重置数据集状态
    void reset();

    // 前向传播,加载一个batch的数据
    // batch_size: 批次大小
    // is_train: 是否为训练模式
    void forward(int batch_size, bool is_train);
    
    // 检查是否还有下一批数据
    bool has_next(bool is_train);

    // 获取图像高度
    int get_height() { return this->height; }
    // 获取图像宽度
    int get_width() { return this->width; }
    // 获取标签数据
    Storage* get_label() { return this->output_label.get();}

    // 打印图像数据(用于调试)
    void print_im();

private:
    unsigned int reverse_int(unsigned int i);  // 大端序转换函数
    
    // 读取MNIST图像文件
    void read_images(std::string file_name, std::vector<std::vector<float>>& output);
    // 读取MNIST标签文件
    void read_labels(std::string file_name, std::vector<unsigned char>& output);

    std::vector<std::vector<float>> train_data;    // 训练数据
    std::vector<unsigned char> train_label;        // 训练标签
    int train_data_index;                          // 训练数据当前索引

    std::vector<std::vector<float>> test_data;     // 测试数据
    std::vector<unsigned char> test_label;         // 测试标签
    int test_data_index;                           // 测试数据当前索引

    int height;                                    // 图像高度
    int width;                                     // 图像宽度
    bool shuffle;                                  // 是否打乱数据标志
    std::unique_ptr<Storage> output_label;         // 输出标签存储

    // 使用CUDA固定内存的训练数据缓冲区
    thrust::host_vector<float, thrust::system::cuda::experimental::pinned_allocator<float>> train_data_buffer;
    // 使用CUDA固定内存的测试数据缓冲区
    thrust::host_vector<float, thrust::system::cuda::experimental::pinned_allocator<float>> test_data_buffer;
};
