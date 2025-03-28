#pragma once

#include <mnist.cuh>
#include <distributed.cuh>
#include <memory>

class DistributedTrainer {
public:
    DistributedTrainer(std::string mnist_data_path, float learning_rate, float l2, float beta);
    
    // 初始化分布式环境
    bool initialize(int rank, int world_size);
    
    // 分布式训练
    void train(int epochs, int batch_size);
    
    // 分布式测试
    void test(int batch_size);
    
private:
    // 训练一个批次
    void train_batch(int batch_size);
    
    // 在所有GPU上同步参数
    void sync_parameters();
    
    // 本地模型
    std::unique_ptr<MNIST> model;
};