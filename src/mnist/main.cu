#include <distributed_trainer.cuh>
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    // 初始化MPI（实际环境中用于进程间通信）
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // 只在rank 0上打印信息
    if (rank == 0) {
        std::cout << "启动分布式训练，进程数: " << world_size << std::endl;
    }
    
    // 创建分布式训练器
    DistributedTrainer trainer("./mnist_data", 0.003, 0.0001, 0.99);
    
    // 初始化分布式环境
    if (!trainer.initialize(rank, world_size)) {
        std::cerr << "进程 " << rank << " 初始化分布式环境失败!" << std::endl;
        MPI_Finalize();
        return 1;
    }
    
    // 等待所有进程初始化完成
    MPI_Barrier(MPI_COMM_WORLD);
    
    // 开始训练
    trainer.train(50, 128);  // 每个进程实际处理 128/world_size 个样本
    
    // 清理分布式环境
    destroy_distributed();
    
    MPI_Finalize();
    return 0;
}

