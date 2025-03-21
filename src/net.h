#pragma once
#include "activation_function.h"
#include "linalg.h"
#include "layer.h"
#include "loss_function.h"
#include "dataloader.h"
#include <vector>

namespace network {

/// НЕ ЧИТАЙ
class Net {
    Net(DataLoader&& dl, LossFunc::Name name);
    void AddLayer(In input_size, Out output_size, ActivationFunc::Name name);
    std::vector<Matrix> Forward(const Matrix& batch);
    void Backward();
    double RunTest();

private:
    DataLoader dl_;
    LossFunc loss_func_;
    std::vector<Layer> layers_;
};
}  // namespace network