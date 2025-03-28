#pragma once

#include <nccl.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>

// 前向声明Storage类
class Storage;

// 分布式环境配置
struct DistributedConfig {
    int world_size;     // 总进程数
    int rank;           // 当前进程编号
    int local_rank;     // 当前进程在节点内的编号
    ncclComm_t comm;    // NCCL通信器
    cudaStream_t stream; // CUDA流用于异步通信
    
    static DistributedConfig& get_instance() {
        static DistributedConfig instance;
        return instance;
    }
};

// 初始化分布式环境
bool initialize_distributed(int rank, int world_size);

// 销毁分布式环境
void destroy_distributed();

// 分布式操作扩展
namespace distributed {
    // 梯度全归约（所有GPU求和）
    void all_reduce(Storage* storage);
    
    // 广播（从一个GPU广播到所有GPU）
    void broadcast(Storage* storage, int root);
    
    // 屏障同步所有进程
    void barrier();
}