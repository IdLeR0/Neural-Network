
#include <iostream>
#include <chrono>
#include <iomanip>
#include "net.h"
#include "activation_function.h"
#include "adam_optimizer.h"
#include "layer.h"

namespace network {
Net::Net(const LayerSizes& layer_sizes, const ActivationFunctions& activation_functions,
         const Params& layer_params) {
    assert(!layer_sizes.empty() && "layer_sizes empty");
    assert(layer_sizes.size() - activation_functions.size() == 1 && "invalid parametrs");
    assert(layer_params.empty() || activation_functions.size() == layer_params.size() &&
                                       "invalid sizes of layer_params or activation_functions");
    if (layer_params.empty()) {
        for (int i = 1; i < layer_sizes.size(); ++i) {
            AddLayer(In{layer_sizes[i - 1]}, Out{layer_sizes[i]}, activation_functions[i - 1]);
        }

    } else {
        CheckParams(layer_sizes, activation_functions, layer_params);
        for (int i = 0; i < layer_params.size(); ++i) {
            AddLayer(layer_params[i].weights, layer_params[i].bias, activation_functions[i]);
        }
    }
}

Net::ComputedBatches Net::Forward(Matrix& cur_input) const {
    std::vector<Matrix> inputs;
    inputs.push_back(cur_input);
    for (const Layer& layer : layers_) {
        cur_input = layer.Forward(cur_input);
        inputs.push_back(cur_input);
    }
    return inputs;
}

Matrix Net::Evaluate(const Matrix& batch) const {
    Matrix cur_batch = batch;
    for (const Layer& layer : layers_) {
        cur_batch = layer.Forward(cur_batch);
    }
    return cur_batch;
}

void Net::Train(DataLoader& dl, LossFunc::Name name, int batch_size, int num_epochs, Info info,
                double start_learning_rate, double beta1, double beta2) {
    assert(!layers_.empty() && "empty layers");
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<AdamOptimizer> optimizers;
    LossFunc loss_func(name);
    for (int i = 0; i < layers_.size(); ++i) {
        Index rows = layers_[i].GetWeightsRows();
        Index cols = layers_[i].GetWeightsCols();
        optimizers.emplace_back(Rows{rows}, Cols{cols}, start_learning_rate, beta1, beta2);
    }
    if (info == Info::On) {

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            auto start = std::chrono::high_resolution_clock::now();

            std::cout << "Epoch" << " " << epoch + 1 << std::endl;

            dl.ShuffleData();
            for (Data& train_batch : dl.Batches(batch_size)) {
                TrainBatch(train_batch, optimizers, loss_func);
            }
            auto stop = std::chrono::high_resolution_clock::now();
            auto time = stop - start;
            auto hours = std::chrono::duration_cast<std::chrono::hours>(time);
            auto minutes =
                std::chrono::duration_cast<std::chrono::minutes>(time % std::chrono::hours(1));
            auto seconds =
                std::chrono::duration_cast<std::chrono::seconds>(time % std::chrono::minutes(1));

            std::cout << "Epoch time:" << " " << std::setfill('0') << std::setw(2) << hours.count()
                      << ":" << std::setw(2) << minutes.count() << ":" << std::setw(2)
                      << seconds.count() << "\n";
        }
        return;
    }
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        dl.ShuffleData();
        for (Data& train_batch : dl.Batches(batch_size)) {
            TrainBatch(train_batch, optimizers, loss_func);
        }
    }
}
FileWriter& operator<<(FileWriter& out, const Net& net) {
    out << net.layers_;
    return out;
}
FileReader& operator>>(FileReader& in, Net& net) {

    in >> net.layers_;
    return in;
}

void Net::AddLayer(In input_size, Out output_size, ActivationFunc::Name name) {
    assert((layers_.empty() || layers_.back().GetWeightsRows() == static_cast<Index>(input_size)) &&
           "incorrect layer");
    layers_.emplace_back(input_size, output_size, name);
}
void Net::AddLayer(const Matrix& weights, const Vector& bias, ActivationFunc::Name name) {
    assert(weights.cols() == bias.rows() && "bad params");
    layers_.emplace_back(weights, bias, name);
}

void Net::CheckParams(const LayerSizes& layer_sizes,
                      const ActivationFunctions& activation_functions, const Params& layer_params) {
    assert(layer_params.size() == activation_functions.size() && "wrong paramers");
    for (int i = 0; i < layer_sizes.size() - 1; ++i) {
        assert(layer_sizes[i] == layer_params[i].weights.cols() && "wrong params");
        assert(layer_sizes[i + 1] == layer_params[i].weights.rows() && "wrong params");
        assert(layer_sizes[i + 1] == layer_params[i].bias.size());
    }
}

void Net::TrainBatch(Data& data, std::vector<AdamOptimizer>& optimizers, LossFunc& loss_func) {
    assert(!layers_.empty() && "no layers");
    assert(layers_.size() == optimizers.size() && "different sizes of optimizers and layers");
    ComputedBatches computed_batces = Forward(data.input);
    assert(computed_batces.size() - layers_.size() == 1 &&
           "invalide size of layers_ or computed_batches");
    Matrix cur_gradient = loss_func.GetGradient(computed_batces.back(), data.output);
    for (int i = layers_.size() - 1; i >= 0; --i) {
        WeightsBiasGradient grad =
            layers_[i].GetWeightsBiasGradient(computed_batces[i], cur_gradient);
        optimizers[i].GetCorrection(grad.weights, grad.bias);
        cur_gradient = layers_[i].Backward(computed_batces[i], cur_gradient);
        layers_[i].UpdateWeights(grad.weights);
        layers_[i].UpdateBias(grad.bias);
    }
}
}  // namespace network