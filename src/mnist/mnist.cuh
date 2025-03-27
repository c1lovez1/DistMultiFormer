#pragma once

// 包含必要的头文件
#include <iostream>              // 输入输出流
#include <memory>               // 智能指针
#include <string>               // 字符串
#include <unordered_map>        // 哈希表

// Thrust相关头文件
#include <thrust/host_vector.h>   // Thrust主机向量
#include <thrust/system/cuda/experimental/pinned_allocator.h>  // CUDA固定内存分配器

// 自定义神经网络组件头文件
#include <blas.cuh>              // BLAS运算库
#include <conv.cuh>              // 卷积层
#include <dataset.cuh>           // 数据集加载
#include <flatten.cuh>           // 展平层
#include <linear.cuh>            // 全连接层
#include <max_pool.cuh>          // 最大池化层
#include <nll_loss.cuh>          // 负对数似然损失
#include <relu.cuh>              // ReLU激活函数
#include <rmsprop.cuh>           // RMSProp优化器
#include <softmax.cuh>           // Softmax激活函数
#include <storage.cuh>           // 数据存储

// MNIST模型类定义
class MNIST {
public:
    // 构造函数
    // mnist_data_path: MNIST数据集路径
    // learning_rate: 学习率
    // l2: L2正则化系数
    // beta: RMSProp的衰减系数
    explicit MNIST(std::string mnist_data_path, float learning_rate, float l2, float beta);

    // 训练模型
    // epochs: 训练轮数
    // batch_size: 批次大小
    void train(int epochs, int batch_size);
    
    // 测试模型
    // batch_size: 批次大小
    void test(int batch_size);

private:
    // 前向传播函数
    void forward(int batch_size, bool is_train);  
    // 反向传播函数
    void backward();                              

    // 计算Top1准确率
    // probs: 预测概率
    // cls_size: 类别数量
    // labels: 真实标签
    std::pair<int, int> top1_accuracy(
        const thrust::host_vector<float, thrust::system::cuda::experimental::pinned_allocator<float>>& probs,
        int cls_size,
        const thrust::host_vector<float, thrust::system::cuda::experimental::pinned_allocator<float>>& labels);

    // 网络结构说明:
    // Conv1_5x5     1 * 32      第一卷积层,1输入通道,32输出通道,5x5卷积核
    // MaxPool1_2x2              第一最大池化层,2x2窗口
    // Conv2_5x5     32 * 64     第二卷积层,32输入通道,64输出通道,5x5卷积核
    // MaxPool2_2x2              第二最大池化层,2x2窗口
    // Conv3_3x3     64 * 128    第三卷积层,64输入通道,128输出通道,3x3卷积核
    // FC1           (128*2*2)*128  第一全连接层,输入128*2*2,输出128
    // FC2           128 * 10       第二全连接层,输入128,输出10
    // LogSoftMax                   对数Softmax层
    // NLLLoss                      负对数似然损失层

    // 模型组件声明
    std::unique_ptr<RMSProp> rmsprop;      // RMSProp优化器
    std::unique_ptr<DataSet> dataset;       // 数据集加载器

    // 第一个卷积块
    std::unique_ptr<Conv> conv1;            // 第一卷积层
    std::unique_ptr<ReLU> conv1_relu;       // 第一ReLU激活
    std::unique_ptr<MaxPool> max_pool1;     // 第一最大池化

    // 第二个卷积块
    std::unique_ptr<Conv> conv2;            // 第二卷积层
    std::unique_ptr<ReLU> conv2_relu;       // 第二ReLU激活
    std::unique_ptr<MaxPool> max_pool2;     // 第二最大池化

    // 第三个卷积块
    std::unique_ptr<Conv> conv3;            // 第三卷积层
    std::unique_ptr<ReLU> conv3_relu;       // 第三ReLU激活
    std::unique_ptr<Flatten> flatten;       // 展平层

    // 全连接块
    std::unique_ptr<Linear> fc1;            // 第一全连接层
    std::unique_ptr<ReLU> fc1_relu;         // 第一全连接ReLU激活

    std::unique_ptr<Linear> fc2;            // 第二全连接层
    std::unique_ptr<ReLU> fc2_relu;         // 第二全连接ReLU激活
    std::unique_ptr<LogSoftmax> log_softmax;// 对数Softmax层
    std::unique_ptr<NLLLoss> nll_loss;      // 负对数似然损失层
};



