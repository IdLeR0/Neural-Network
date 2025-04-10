#pragma once
#include <vector>
#include "file_reader_writer.h"
#include "linalg.h"
#include "activation_function.h"
#include "layer.h"
#include "loss_function.h"
#include "dataloader.h"
#include "adam_optimizer.h"

namespace network {
namespace details {

struct OptimizersParams {
    AdamOptimizer weights;
    AdamOptimizer bias;
};
}  // namespace details
struct LayerParams {
    Matrix weights;
    Vector bias;
};
enum class Info { On, Off };

class Net {
    // using Layers = std::vector<Layer>;
    // using ComputedBatches = std::vector<Matrix>;
    // using Optimizers = std::vector<details::OptimizersParams>;
    // using LayerSizes = std::vector<Index>;
    // using ActivationFunctions = std::vector<ActivationFunc::Name>;

public:
    using Layers = std::vector<Layer>;
    using ComputedBatches = std::vector<Matrix>;
    using Optimizers = std::vector<details::OptimizersParams>;
    using LayerSizes = std::vector<Index>;
    using ActivationFunctions = std::vector<ActivationFunc::Name>;
    using Params = std::vector<LayerParams>;
    Net() = default;
    Net(const LayerSizes& layer_sizes, const ActivationFunctions& activation_functions,
        const Params& layer_params = {});

    Matrix Evaluate(const Matrix& batch) const;
    void Train(DataLoader& dl, LossFunc::Name name, int batch_size, int num_epochs, Info info,
               double start_learning_rate = kDefaultStartLearningRate, double beta1 = kDefaultBeta1,
               double beta2 = kDefaultBeta2);
    friend FileWriter& operator<<(FileWriter& out, const Net& net);
    friend FileReader& operator>>(FileReader& in, Net& net);
    bool operator==(const Net& other) const = default;

private:
    ComputedBatches Forward(Matrix& batch) const;
    void AddLayer(In input_size, Out output_size, ActivationFunc::Name name);
    void AddLayer(const Matrix& weights, const Vector& bias, ActivationFunc::Name name);
    void CheckParams(const LayerSizes& layer_sizes, const ActivationFunctions& activation_functions,
                     const Params& layer_params);
    void TrainBatch(Data& data, std::vector<AdamOptimizer>& optimizers, LossFunc& loss_func);
    static constexpr double kDefaultStartLearningRate = 3e-4;
    static constexpr double kDefaultBeta1 = 0.9;
    static constexpr double kDefaultBeta2 = 0.99;

    Layers layers_;
};
}  // namespace network