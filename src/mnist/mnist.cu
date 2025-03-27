#include <mnist.cuh>
#include <cfloat> // 用于FLT_MAX常量
#include <thrust/host_vector.h>  // Thrust主机向量
#include <thrust/device_vector.h> // Thrust设备向量
#include <thrust/system/cuda/experimental/pinned_allocator.h> // CUDA固定内存分配器

// MNIST类构造函数
// mnist_data_path: MNIST数据集路径
// learning_rate: 学习率
// l2: L2正则化系数
// beta: RMSProp优化器的衰减系数
MNIST::MNIST(std::string mnist_data_path, float learning_rate, float l2, float beta) {
    // 初始化数据集,启用数据打乱
    dataset.reset(new DataSet(mnist_data_path, true));

    // 第一个卷积块: Conv1(1->32) + ReLU + MaxPool
    conv1.reset(new Conv(28, 28, 1, 32, 5, 5, 0, 0, 1, 1, true));  // 输入28x28x1,输出24x24x32
    conv1_relu.reset(new ReLU(true));// 激活函数 原地操作
    max_pool1.reset(new MaxPool(2, 2, 0, 0, 2, 2));  // 输出12x12x32

    // 第二个卷积块: Conv2(32->64) + ReLU + MaxPool
    conv2.reset(new Conv(12, 12, 32, 64, 5, 5, 0, 0, 1, 1, true));  // 输入12x12x32,输出8x8x64
    conv2_relu.reset(new ReLU(true));// 激活函数 原地操作
    max_pool2.reset(new MaxPool(2, 2, 0, 0, 2, 2));  // 输出4x4x64

    // 第三个卷积块: Conv3(64->128) + ReLU
    conv3.reset(new Conv(4, 4, 64, 128, 3, 3, 0, 0, 1, 1, true));  // 输入4x4x64,输出2x2x128
    conv3_relu.reset(new ReLU(true));// 激活函数 原地操作

    // 全连接层部分
    flatten.reset(new Flatten(true));  // 将2x2x128展平为512
    fc1.reset(new Linear(128 * 2 * 2, 128, true));  // 512->128 使用偏置项
    fc1_relu.reset(new ReLU(true));// 激活函数 原地操作

    fc2.reset(new Linear(128, 10, true));  // 128->10(类别数) 使用偏置项    
    fc2_relu.reset(new ReLU(true));// 激活函数 原地操作 

    // 输出层
    log_softmax.reset(new LogSoftmax(1));  // 对数Softmax 在维度1上进行计算 [N, C_out]
    nll_loss.reset(new NLLLoss());  // 负对数似然损失 

    // 构建网络前向传播图,按顺序连接各层
    dataset->connect(*conv1)
            .connect(*conv1_relu)
            .connect(*max_pool1)
            .connect(*conv2)
            .connect(*conv2_relu)
            .connect(*max_pool2)
            .connect(*conv3)
            .connect(*conv3_relu)
            .connect(*flatten)
            .connect(*fc1)
            .connect(*fc1_relu)
            .connect(*fc2)
            .connect(*fc2_relu)
            .connect(*log_softmax)
            .connect(*nll_loss);

    // 初始化RMSProp优化器并注册需要优化的参数
    rmsprop.reset(new RMSProp(learning_rate, l2, beta));
    rmsprop->regist(conv1->parameters());  // 注册Conv1的参数
    rmsprop->regist(conv2->parameters());  // 注册Conv2的参数
    rmsprop->regist(conv3->parameters());  // 注册Conv3的参数
    rmsprop->regist(fc1->parameters());    // 注册FC1的参数
    rmsprop->regist(fc2->parameters());    // 注册FC2的参数
}

// MNIST模型训练函数
// epochs: 训练轮数
// batch_size: 每批次样本数量
void MNIST::train(int epochs, int batch_size) {
    // 遍历训练轮数
    for (int epoch = 0; epoch < epochs; epoch++) {
        int idx = 1;  // 批次计数器

        // 当数据集还有下一批训练数据时继续训练
        while (dataset->has_next(true)) {
            forward(batch_size, true);   // 前向传播
            backward();                  // 反向传播计算梯度
            rmsprop->step();            // 使用RMSProp优化器更新参数

            // 每10个批次打印一次训练状态
            if (idx % 10 == 0) {
                // 获取当前批次的损失值
                float loss = this->nll_loss->get_output()->get_data()[0];
                // 计算当前批次的Top1准确率
                // 返回值: pair<正确预测数量, 总样本数量>   
                auto acc = top1_accuracy(this->log_softmax->get_output()->get_data(),
                                         10, this->dataset->get_label()->get_data());

                // 打印训练信息:当前轮数、批次、损失值和准确率
                std::cout << "Epoch: " << epoch << ", Batch: " << idx
                          << ", NLLLoss: " << loss
                          << ", Train Accuracy: " << (float(acc.first) / acc.second)
                          << std::endl;
            }
            ++idx;  // 批次计数器递增
        }

        test(batch_size);     // 每轮结束后在测试集上评估模型
        dataset->reset();     // 重置数据集状态,准备下一轮训练
    }
}

// MNIST模型测试函数
// batch_size: 每批次测试样本数量
void MNIST::test(int batch_size) {
    int idx = 1;          // 批次计数器
    int count = 0;        // 累计正确预测的样本数
    int total = 0;        // 累计总样本数

    // 当测试集还有下一批数据时继续测试
    while (dataset->has_next(false)) {
        forward(batch_size, false);  // 前向传播(不进行训练)
        
        // 计算当前批次的Top1准确率
        auto acc = top1_accuracy(this->log_softmax->get_output()->get_data(), 10,
                                 this->dataset->get_label()->get_data());
        
        // 每10个批次打印一次测试状态
        if (idx % 10 == 0) {
            std::cout << "Batch: " << idx
                      << ", Test Accuracy: " << (float(acc.first) / acc.second)
                      << std::endl;
        }

        // 累加正确预测数和总样本数
        count += acc.first;    // 累加当前批次正确预测数
        total += acc.second;   // 累加当前批次总样本数
        ++idx;                 // 批次计数器递增
    }

    // 打印整个测试集的总体准确率
    std::cout << "Total Accuracy: " << (float(count) / total) << std::endl;
}

// MNIST模型的前向传播函数
// batch_size: 每批次样本数量
// is_train: 是否为训练模式
void MNIST::forward(int batch_size, bool is_train) {
    // 从数据集获取下一批数据
    dataset->forward(batch_size, is_train);
    // 获取标签数据
    const Storage* labels = dataset->get_label();

    // 第一个卷积块:卷积->ReLU激活->最大池化
    conv1->forward();         // 第一层卷积
    conv1_relu->forward();    // ReLU激活函数
    max_pool1->forward();     // 最大池化层

    // 第二个卷积块:卷积->ReLU激活->最大池化  
    conv2->forward();         // 第二层卷积
    conv2_relu->forward();    // ReLU激活函数
    max_pool2->forward();     // 最大池化层

    // 第三个卷积块:卷积->ReLU激活
    conv3->forward();         // 第三层卷积
    conv3_relu->forward();    // ReLU激活函数

    // 全连接分类器部分
    flatten->forward();       // 展平层,将特征图转换为一维向量
    fc1->forward();          // 第一个全连接层
    fc1_relu->forward();     // ReLU激活函数

    fc2->forward();          // 第二个全连接层
    fc2_relu->forward();     // ReLU激活函数

    // 输出层
    log_softmax->forward();   // 对数Softmax层,用于多分类

    // 仅在训练模式下计算损失
    if (is_train) nll_loss->forward(labels);  // 负对数似然损失
}

// MNIST模型的反向传播函数
// 按照网络结构从后向前依次计算每一层的梯度
void MNIST::backward() {
    // 从损失函数开始反向传播
    nll_loss->backward();      // 计算负对数似然损失的梯度
    log_softmax->backward();   // 计算对数Softmax层的梯度

    // 全连接分类器部分的反向传播
    fc2_relu->backward();      // 第二个全连接层的ReLU激活函数梯度
    fc2->backward();           // 第二个全连接层的梯度

    fc1_relu->backward();      // 第一个全连接层的ReLU激活函数梯度
    fc1->backward();           // 第一个全连接层的梯度
    flatten->backward();       // 展平层的梯度

    // 第三个卷积块的反向传播
    conv3_relu->backward();    // 第三层卷积的ReLU激活函数梯度
    conv3->backward();         // 第三层卷积的梯度

    // 第二个卷积块的反向传播
    max_pool2->backward();     // 第二个最大池化层的梯度
    conv2_relu->backward();    // 第二层卷积的ReLU激活函数梯度
    conv2->backward();         // 第二层卷积的梯度

    // 第一个卷积块的反向传播
    max_pool1->backward();     // 第一个最大池化层的梯度
    conv1_relu->backward();    // 第一层卷积的ReLU激活函数梯度
    conv1->backward();         // 第一层卷积的梯度
}

// 计算Top1准确率的函数
// 参数说明:
// probs: 模型预测的概率分布向量(使用固定内存分配)
// cls_size: 类别数量
// labels: 真实标签的one-hot编码向量(使用固定内存分配)
// 返回值: pair<正确预测数量, 总样本数量>
std::pair<int, int> MNIST::top1_accuracy(
    const thrust::host_vector<
        float, thrust::system::cuda::experimental::pinned_allocator<float>>&
        probs,
    int cls_size, 
    const thrust::host_vector<
        float, thrust::system::cuda::experimental::pinned_allocator<float>>&
        labels) {
    int count = 0;  // 正确预测的样本数量
    int size = labels.size() / cls_size;  // 总样本数量(标签总长度/类别数)
    
    // 遍历每个样本
    for (int i = 0; i < size; i++) {
        // 用于存储预测概率的最大值及其位置
        int max_pos = -1;
        float max_value = -FLT_MAX;
        // 用于存储真实标签的最大值(1)及其位置
        int max_pos2 = -1;
        float max_value2 = -FLT_MAX;

        // 遍历每个类别
        for (int j = 0; j < cls_size; j++) {
            int index = i * cls_size + j;  // 计算在展平向量中的索引位置
            // 更新预测概率的最大值和位置
            if (probs[index] > max_value) {
                max_value = probs[index];
                max_pos = j;
            }
            // 更新真实标签的最大值和位置
            // 标签是one-hot编码,最大值为1,位置为真实类别，其他位置为0  
            if (labels[index] > max_value2) {
                max_value2 = labels[index];
                max_pos2 = j;
            }
        }
        // 如果预测类别与真实类别相同,正确计数加1
        if (max_pos == max_pos2) ++count;
    }
    // 返回正确预测数量和总样本数量的配对
    return {count, size};
}
