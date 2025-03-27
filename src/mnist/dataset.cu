// 包含必要的头文件
#include <dataset.cuh>     // 数据集类定义
#include <utils.cuh>       // 工具函数
#include <thrust/fill.h>   // Thrust填充函数
#include <vector>          // 向量容器
#include <algorithm>       // 算法库
#include <chrono>         // 时间相关
#include <fstream>        // 文件操作
#include <random>         // 随机数
#include <iostream>       // 输入输出流
#include <cfloat>         // 浮点数限制,用于FLT_MAX

/**
 * @brief MNIST数据集类的构造函数
 * @param mnist_data_path MNIST数据集的路径
 * @param shuffle 是否需要打乱数据的标志
 */
DataSet::DataSet(std::string mnist_data_path, bool shuffle)
    : shuffle(shuffle),           // 初始化是否打乱数据的标志
      train_data_index(0),       // 初始化训练数据索引
      test_data_index(0) {       // 初始化测试数据索引
      
  // 注释掉的是原始MNIST数据集文件名格式
  // // 读取训练数据
  // this->read_images(mnist_data_path + "/train-images-idx3-ubyte",
  //                   this->train_data);
  // this->read_labels(mnist_data_path + "/train-labels-idx1-ubyte",
  //                   this->train_label);
  // // 读取测试数据
  // this->read_images(mnist_data_path + "/t10k-images-idx3-ubyte",
  //                   this->test_data);
  // this->read_labels(mnist_data_path + "/t10k-labels-idx1-ubyte",
  //                   this->test_label);

  // 读取训练集图像和标签
  this->read_images(mnist_data_path + "/train-images.idx3-ubyte",
                    this->train_data);
  this->read_labels(mnist_data_path + "/train-labels.idx1-ubyte",
                    this->train_label);
                    
  // 读取测试集图像和标签
  this->read_images(mnist_data_path + "/t10k-images.idx3-ubyte",
                    this->test_data);
  this->read_labels(mnist_data_path + "/t10k-labels.idx1-ubyte",
                    this->test_label);
}

/**
 * @brief 重置数据集状态的函数
 * 该函数用于:
 * 1. 重置训练和测试数据的索引
 * 2. 如果启用了shuffle,则对训练数据进行随机打乱
 */
void DataSet::reset() {
  // 重置训练数据索引为0
  this->train_data_index = 0;
  // 重置测试数据索引为0  
  this->test_data_index = 0;

  // 如果启用了数据打乱
  if (shuffle) {
    // 生成随机种子
    // 使用当前时间戳对1234取模,保证每次运行时种子相同
    unsigned int seed =
        std::chrono::system_clock::now().time_since_epoch().count() % 1234;

    // 使用相同的随机种子打乱训练数据
    std::shuffle(this->train_data.begin(), this->train_data.end(),
                 std::default_random_engine(seed));
    // 使用相同的随机种子打乱训练标签
    // 保证数据和标签的对应关系不变
    std::shuffle(this->train_label.begin(), this->train_label.end(),
                 std::default_random_engine(seed));
  }
}

/**
 * @brief 前向传播函数,用于加载一个batch的数据
 * @param batch_size 每个批次的样本数量
 * @param is_train 是否为训练模式
 * 该函数主要功能:
 * 1. 根据is_train加载训练集或测试集数据
 * 2. 初始化设备内存并分配空间
 * 3. 将数据从主机内存拷贝到设备内存
 * 4. 生成one-hot编码的标签
 */
void DataSet::forward(int batch_size, bool is_train) {
  if (is_train) {  // 训练模式
    // 计算当前批次的起始和结束索引
    int start = this->train_data_index;
    int end = std::min(this->train_data_index + batch_size,
                       (int)this->train_data.size());
    this->train_data_index = end;
    int size = end - start;  // 实际批次大小

    // 初始化设备内存
    std::vector<int> output_shape{size, 1, this->height, this->width};  // 输出数据形状为[batch_size, channels=1, height, width]
    std::vector<int> output_label_shape{size, 10};  // 标签形状(10个类别)为[batch_size, num_classes=10]
    INIT_STORAGE(this->output, output_shape);  // 初始化输出存储,形状为[batch_size, 1, height, width]
    INIT_STORAGE(this->output_label, output_label_shape);  // 初始化标签存储,形状为[batch_size, 10]
    // 将标签存储初始化为0
    thrust::fill(this->output_label->get_data().begin(),
                 this->output_label->get_data().end(), 0);

    // 计算步长
    int im_stride = 1 * this->height * this->width;  // 图像数据步长 每次一张图像
    int one_hot_stride = 10;  // one-hot编码步长 

    // 创建固定内存缓冲区用于数据传输
    thrust::host_vector<
        float, thrust::system::cuda::experimental::pinned_allocator<float>>
        train_data_buffer;
    train_data_buffer.reserve(size * im_stride);

    // 拷贝数据到设备内存并生成one-hot标签
    for (int i = start; i < end; i++) {
      // 将训练数据拷贝到缓冲区
      train_data_buffer.insert(train_data_buffer.end(),
                               this->train_data[i].begin(),
                               this->train_data[i].end());
      // 设置对应类别的one-hot标签为1
      this->output_label
          ->get_data()[(i - start) * one_hot_stride + this->train_label[i]] = 1;
    }
    // 将缓冲区数据拷贝到输出存储
    this->output->get_data() = train_data_buffer;

  } else {  // 测试模式
    // 计算当前批次的起始和结束索引
    int start = this->test_data_index;
    int end = std::min(this->test_data_index + batch_size,
                       (int)this->test_data.size());
    this->test_data_index = end;
    int size = end - start;  // 实际批次大小

    // 初始化设备内存
    std::vector<int> output_shape{size, 1, this->height, this->width};  // 输出数据形状
    std::vector<int> output_label_shape{size, 10};  // 标签形状(10个类别)
    INIT_STORAGE(this->output, output_shape);  // 初始化输出存储
    INIT_STORAGE(this->output_label, output_label_shape);  // 初始化标签存储
    // 将标签存储初始化为0
    thrust::fill(this->output_label->get_data().begin(),
                 this->output_label->get_data().end(), 0);

    // 计算步长
    int im_stride = 1 * this->height * this->width;  // 图像数据步长
    int one_hot_stride = 10;  // one-hot编码步长

    // 创建固定内存缓冲区用于数据传输
    thrust::host_vector<
        float, thrust::system::cuda::experimental::pinned_allocator<float>>
        test_data_buffer;
    test_data_buffer.reserve(size * im_stride);

    // 拷贝数据到设备内存并生成one-hot标签
    for (int i = start; i < end; i++) {
      // 将测试数据拷贝到缓冲区
      test_data_buffer.insert(test_data_buffer.end(),
                              this->test_data[i].begin(),
                              this->test_data[i].end());
      // 设置对应类别的one-hot标签为1
      this->output_label
          ->get_data()[(i - start) * one_hot_stride + this->test_label[i]] = 1;
    }
    // 将缓冲区数据拷贝到输出存储
    this->output->get_data() = test_data_buffer;
  }
}

// 检查是否还有下一批数据可以处理
// 参数:
//   is_train: 布尔值,表示是否为训练模式
// 返回值:
//   如果还有未处理的数据返回true,否则返回false
bool DataSet::has_next(bool is_train) {
  if (is_train) {
    // 训练模式:检查训练数据索引是否小于训练数据总量
    return this->train_data_index < this->train_data.size();
  } else {
    // 测试模式:检查测试数据索引是否小于测试数据总量
    return this->test_data_index < this->test_data.size();
  }
}

// 打印当前批次的图像数据和对应标签
// 用于调试和可视化数据
void DataSet::print_im() {
  // 获取当前批次的样本数量
  int size = this->output->get_shape()[0];
  // 计算每个图像的数据步长(通道数*高度*宽度)
  int im_stride = 1 * height * width;

  // 遍历每个样本
  for (int k = 0; k < size; k++) {
    // 找出one-hot标签中值最大的位置,即预测的类别
    int max_pos = -1;
    float max_value = -FLT_MAX;
    for (int i = 0; i < 10; i++) {
      float val = this->output_label->get_data()[k * 10 + i];
      if (val > max_value) {
        max_value = val;
        max_pos = i;
      }
    }

    // 打印预测的类别
    std::cout << max_pos << std::endl;
    // 获取图像数据的引用
    auto& data = this->output->get_data();
    // 按行列打印图像,大于0的像素点用"*"表示,否则用空格表示
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        std::cout << (data[k * im_stride + i * width + j] > 0 ? "* " : "  ");
      }
      std::cout << std::endl;
    }
  }
}

// 将大端序整数转换为小端序
// 参数:
//   i: 输入的大端序整数
// 返回值:
//   转换后的小端序整数
unsigned int DataSet::reverse_int(unsigned int i) {
  unsigned char ch1, ch2, ch3, ch4;
  // 提取每个字节
  ch1 = i & 255;           // 最低字节
  ch2 = (i >> 8) & 255;    // 次低字节
  ch3 = (i >> 16) & 255;   // 次高字节
  ch4 = (i >> 24) & 255;   // 最高字节
  // 重新组合字节顺序
  return ((unsigned int)ch1 << 24) + ((unsigned int)ch2 << 16) +
         ((unsigned int)ch3 << 8) + ch4;
}

// 读取MNIST图像文件并解析
// 参数:
//   file_name: 图像文件路径
//   output: 存储解析后图像数据的向量
void DataSet::read_images(std::string file_name,
                          std::vector<std::vector<float>>& output) {
  // 以二进制模式打开文件
  std::ifstream file(file_name, std::ios::binary);
  if (file.is_open()) {
    // 文件头信息
    unsigned int magic_number = 0;      // 魔数,用于验证文件格式
    unsigned int number_of_images = 0;  // 图像总数
    unsigned int n_rows = 0;            // 图像高度
    unsigned int n_cols = 0;            // 图像宽度

    // 读取文件头信息
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_images, sizeof(number_of_images));
    file.read((char*)&n_rows, sizeof(n_rows));
    file.read((char*)&n_cols, sizeof(n_cols));

    // 转换字节序
    magic_number = this->reverse_int(magic_number);
    number_of_images = this->reverse_int(number_of_images);
    n_rows = this->reverse_int(n_rows);
    n_cols = this->reverse_int(n_cols);

    // 打印文件信息
    std::cout << file_name << std::endl;
    std::cout << "magic number = " << magic_number << std::endl;
    std::cout << "number of images = " << number_of_images << std::endl;
    std::cout << "rows = " << n_rows << std::endl;
    std::cout << "cols = " << n_cols << std::endl;

    // 设置图像尺寸
    this->height = n_rows;
    this->width = n_cols;

    // 创建临时缓冲区
    std::vector<unsigned char> image(n_rows * n_cols);    // 原始图像数据
    std::vector<float> normalized_image(n_rows * n_cols); // 归一化后的图像数据

    // 读取并处理每张图像
    for (int i = 0; i < number_of_images; i++) {
      // 读取原始图像数据
      file.read((char*)&image[0], sizeof(unsigned char) * n_rows * n_cols);

      // 归一化处理: 将像素值从[0,255]映射到[-0.5,0.5]
      for (int i = 0; i < n_rows * n_cols; i++) {
        normalized_image[i] = (float)image[i] / 255 - 0.5;
      }
      // 保存归一化后的图像
      output.push_back(normalized_image);
    }
  }
}

// 读取MNIST标签文件的函数
// file_name: 标签文件路径
// output: 用于存储读取的标签数据的向量
void DataSet::read_labels(std::string file_name,
                          std::vector<unsigned char>& output) {
  // 以二进制模式打开文件
  std::ifstream file(file_name, std::ios::binary);
  if (file.is_open()) {
    // 定义文件头信息变量
    unsigned int magic_number = 0;        // 魔数,用于验证文件格式 标签文件的魔数是2049
    unsigned int number_of_images = 0;    // 标签总数 （训练集60000，测试集10000）
    
    // 读取文件头信息
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_images, sizeof(number_of_images));
    
    // 转换字节序(从大端序到小端序)
    magic_number = this->reverse_int(magic_number);
    number_of_images = this->reverse_int(number_of_images);
    
    // 打印文件名
    std::cout << file_name << std::endl;
    // 打印文件头信息
    std::cout << "magic number = " << magic_number << std::endl;
    std::cout << "number of images = " << number_of_images << std::endl;

    // 逐个读取标签
    for (int i = 0; i < number_of_images; i++) {
      unsigned char label = 0;  // 临时存储单个标签
      file.read((char*)&label, sizeof(label));  // 读取一个字节的标签数据
      output.push_back(label);  // 将标签添加到输出向量中
    }
  }
}