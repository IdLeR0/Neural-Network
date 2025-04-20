#pragma once
#include <vector>
#include "file_reader_writer.h"
#include "global_usings.h"
#include "activation_function.h"
#include "layer.h"
#include "loss_function.h"
#include "dataloader.h"
#include "adam_optimizer.h"

namespace network {

enum class Info { On, Off };

class Net {
    using Layers = std::vector<Layer>;
    using ComputedBatches = std::vector<Matrix>;
    using Optimizers = std::vector<AdamOptimizer>;
    using LayerSizes = std::vector<Index>;
    using ActivationFunctions = std::vector<ActivationFunc::Name>;
    using Params = std::vector<LayerParams>;

public:
    Net() = default;
    Net(const LayerSizes& layer_sizes, const ActivationFunctions& activation_functions,
        const Params& layer_params = {});

    Matrix Evaluate(const Matrix& batch) const;
    void Train(DataLoader& dl, LossFunc::Name name, Index batch_size, Index num_epochs, Info info,
               DataType learning_rate = kDefaultStartLearningRate, DataType beta1 = kDefaultBeta1,
               DataType beta2 = kDefaultBeta2);
    friend FileWriter& operator<<(FileWriter& out, const Net& net);
    friend FileReader& operator>>(FileReader& in, Net& net);
    bool operator==(const Net& other) const = default;

private:
    ComputedBatches Forward(const Matrix& batch) const;
    void AddLayer(In input_size, Out output_size, ActivationFunc::Name name);
    void AddLayer(const Matrix& weights, const Vector& bias, ActivationFunc::Name name);
    void CheckParams(const LayerSizes& layer_sizes, const ActivationFunctions& activation_functions,
                     const Params& layer_params);
    void TrainBatch(const DataView& data, Optimizers& optimizers, const LossFunc& loss_func);
    static constexpr DataType kDefaultStartLearningRate = 3e-4;
    static constexpr DataType kDefaultBeta1 = 0.9;
    static constexpr DataType kDefaultBeta2 = 0.99;

    Layers layers_;
};
}  // namespace network