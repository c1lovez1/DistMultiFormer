#pragma once

#include <utility>
#include <vector>

#include <storage.cuh>

class Optimizer {
 public:
  virtual void step() = 0;
  virtual void regist(std::vector<std::pair<Storage *, Storage *>> params) = 0;

  virtual void distributed_step() {
    // 首先同步所有梯度
    sync_gradients();
    // 然后执行常规参数更新
    step();
  }

  // 同步所有梯度 新加的需要测试一下
  virtual void sync_gradients() {
    for (auto* grad : grad_list) {
      grad->all_reduce();
    }
  }

 protected:
  std::vector<Storage *> parameter_list;
  std::vector<Storage *> grad_list;
};