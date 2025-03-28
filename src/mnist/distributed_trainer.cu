#include <distributed_trainer.cuh>

DistributedTrainer::DistributedTrainer(std::string mnist_data_path, float learning_rate, float l2, float beta) {
    model.reset(new MNIST(mnist_data_path, learning_rate, l2, beta));
}

bool DistributedTrainer::initialize(int rank, int world_size) {
    if (!initialize_distributed(rank, world_size)) {
        return false;
    }
    
    // 同步所有参数
    sync_parameters();
    
    // 添加参数验证
    distributed::barrier();
    if (rank == 0) {
        std::cout << "分布式环境初始化完成，参数已同步" << std::endl;
    }
    
    return true;
}

void DistributedTrainer::sync_parameters() {
    auto& config = DistributedConfig::get_instance();
    
    // 广播所有模型参数
    auto params = model->get_parameters();
    for (auto& param : params) {
        param->broadcast(0);  // 从0号进程广播
    }
}

void DistributedTrainer::train(int epochs, int batch_size) {
    auto& config = DistributedConfig::get_instance();
    
    // 计算每个GPU处理的批次大小
    int local_batch_size = batch_size / config.world_size;
    // 添加每个epoch的最大批次数
    int max_batches_per_epoch = 60000 / batch_size + 1;
    
    if (config.rank == 0) {
        std::cout << "开始分布式训练, 使用 " << config.world_size 
                  << " 个GPU, 每个处理 " << local_batch_size << " 样本" << std::endl;
    }
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // 重置数据集
        model->reset_dataset();
        
        int batch_idx = 0;
        // 添加批次数量限制
        while (model->has_next_batch(true) && batch_idx < max_batches_per_epoch) {
            // 训练一个批次
            train_batch(local_batch_size);
            
            // 只在主进程上打印信息
            if (config.rank == 0 && batch_idx % 10 == 0) {
                float loss = model->get_loss();
                auto acc = model->get_accuracy();
                std::cout << "Epoch: " << epoch << ", Batch: " << batch_idx
                          << ", Loss: " << loss
                          << ", Acc: " << (float)acc.first / acc.second
                          << std::endl;
            }
            
            batch_idx++;
        }
        
        if (config.rank == 0) {
            std::cout << "第 " << epoch << " 轮训练完成，处理了 " << batch_idx << " 个批次" << std::endl;
        }
        
        // 每个epoch结束后在测试集上评估
        if (config.rank == 0) {
            test(batch_size);
        }
        
        // 同步所有进程
        distributed::barrier();
    }
}

void DistributedTrainer::train_batch(int batch_size) {
    auto& config = DistributedConfig::get_instance();
    
    // 1. 前向传播（每个GPU使用不同数据）
    model->forward(batch_size, true, config.rank);
    
    // 2. 反向传播
    model->backward();
    
    // 3. 同步所有梯度
    auto grads = model->get_gradients();
    for (auto* grad : grads) {
        grad->all_reduce();  // 已经在all_reduce中除以GPU数量
    }
    
    // 4. 更新参数（使用已同步的梯度）
    model->update_parameters();
}

void DistributedTrainer::test(int batch_size) {
    model->test(batch_size);
}