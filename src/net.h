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
    Net(std::vector<int> sizes_of_layers,
        std::vector<ActivationFunc::Name> names_of_activation_functions);
    Matrix Evaluate(const Matrix& batch) const;
    void Backward(DataLoader& dl, LossFunc::Name name, int batch_size, int num_epochs,
                  double start_learning_rate = kDefaultStartLearningRate,
                  double beta1 = kDefaultBeta1, double beta2 = kDefaultBeta2);

private:
    ComputedBatches Forward(Matrix& batch) const;
    void AddLayer(In input_size, Out output_size, ActivationFunc::Name name);
    void TrainBatch(Data& data, std::vector<AdamOptimizer>& optimizers, LossFunc& loss_func);
    static constexpr double kDefaultStartLearningRate = 3e-4;
    static constexpr double kDefaultBeta1 = 0.9;
    static constexpr double kDefaultBeta2 = 0.99;
    Layers layers_;
};
}  // namespace network