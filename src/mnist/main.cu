#include <dataset.cuh>
#include <mnist.cuh>
#include <chrono>
#include <iostream>

#define BATCH_SIZE 128
#define LEARNING_RATE 0.003
#define L2 0.0001
#define EPOCHS 50
#define BETA 0.99

int main() {
  // DataSet dataset("./mnist_data", true);
  // dataset.forward(64, true);
  // dataset.print_im();

  auto cudaStatus = cudaSetDevice(0);
  CHECK_EQ(cudaStatus, cudaSuccess,
           "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

  auto start = std::chrono::high_resolution_clock::now();

  MNIST mnist("./mnist_data", LEARNING_RATE, L2, BETA);
  mnist.train(EPOCHS, BATCH_SIZE);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
  std::cout << "总训练时间: " << duration.count() << " 秒" << std::endl;
}

