#pragma once
#include "linalg.h"
#include "activation_function.h"
#include "layer.h"
#include "loss_function.h"
#include "dataloader.h"
#include "adam_optimizer.h"
#include <vector>

namespace network {
namespace details {

struct OptimizersParams {
    AdamOptimizer weights;
    AdamOptimizer bias;
};
}  // namespace details

class Net {
    using Layers = std::vector<Layer>;
    using ComputedBatches = std::vector<Matrix>;
    using Optimizers = std::vector<details::OptimizersParams>;

public:
    Net(DataLoader&& dl, LossFunc::Name name);
    void AddLayer(In input_size, Out output_size, ActivationFunc::Name name);
    ComputedBatches Forward(const Matrix& batch) const;
    Matrix Evalute(const Matrix& batch) const;
    void Backward(int batch_size, int num_epochs,
                  double start_learning_rate = kDefaultStartLearningRate,
                  double beta1 = kDefaultBeta1, double beta2 = kDefaultBeta2);

private:
    void TrainBatch(const Data& data);
    static constexpr double kDefaultStartLearningRate = 3e-4;
    static constexpr double kDefaultBeta1 = 0.9;
    static constexpr double kDefaultBeta2 = 0.99;
    DataLoader dl_;
    LossFunc loss_func_;
    Layers layers_;
    Optimizers opts_;
};
}  // namespace network