#include <distributed.cuh>
#include <storage.cuh>
#include <blas.cuh>  // 如果需要operator_mul
#include <utils.cuh>
#include <iostream>
#include <mpi.h>


bool initialize_distributed(int rank, int world_size) {
    auto& config = DistributedConfig::get_instance();
    config.rank = rank;
    config.world_size = world_size;
    config.local_rank = rank % 1; // 假设每节点1个GPU，实际应根据环境设置
    
    // 设置当前设备
    CUDA_CHECK(cudaSetDevice(config.local_rank));
    
    // 创建CUDA流
    CUDA_CHECK(cudaStreamCreate(&config.stream));
    
    // 初始化NCCL
    ncclUniqueId id;
    if (rank == 0) {
        // 主进程生成ID
        ncclGetUniqueId(&id);
    }
    
    // 使用MPI广播NCCL ID到所有进程
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // 创建NCCL通信器
    ncclResult_t result = ncclCommInitRank(&config.comm, world_size, id, rank);
    if (result != ncclSuccess) {
        std::cerr << "NCCL初始化失败: " << ncclGetErrorString(result) << std::endl;
        return false;
    }
    
    return true;
}

void destroy_distributed() {
    auto& config = DistributedConfig::get_instance();
    ncclCommDestroy(config.comm);
    cudaStreamDestroy(config.stream);
}

namespace distributed {
    void all_reduce(Storage* storage) {
        auto& config = DistributedConfig::get_instance();
        float* data_ptr = RAW_PTR(storage->get_data());
        size_t count = storage->get_data().size();
        
        // 添加错误处理
        ncclResult_t result = ncclAllReduce(
            data_ptr, data_ptr, count, ncclFloat, ncclSum, 
            config.comm, config.stream
        );
        
        if (result != ncclSuccess) {
            std::cerr << "NCCL AllReduce失败: " << ncclGetErrorString(result) << std::endl;
            // 可以添加更多恢复机制
        }
        
        // 同步流
        cudaStreamSynchronize(config.stream);
        
        // 计算平均值
        if (config.world_size > 1) {
            float scale = 1.0f / config.world_size;
            operator_mul(storage, scale, storage);
        }
    }
    
    void broadcast(Storage* storage, int root) {
        auto& config = DistributedConfig::get_instance();
        float* data_ptr = RAW_PTR(storage->get_data());
        size_t count = storage->get_data().size();
        
        // 执行Broadcast操作
        ncclBroadcast(
            data_ptr, 
            data_ptr, 
            count, 
            ncclFloat, 
            root, 
            config.comm, 
            config.stream
        );
        
        // 同步流
        cudaStreamSynchronize(config.stream);
    }
    
    void barrier() {
        auto& config = DistributedConfig::get_instance();
        ncclAllReduce(nullptr, nullptr, 0, ncclFloat, ncclSum, config.comm, config.stream);
        cudaStreamSynchronize(config.stream);
    }
}