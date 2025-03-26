#include "net.h"
#include "activation_function.h"
#include "adam_optimizer.h"
#include "layer.h"

#include <iostream>
#include <chrono>

namespace network {
Net::Net(std::vector<int> sizes_of_layers,
         std::vector<ActivationFunc::Name> names_of_activation_functions) {
    assert(!sizes_of_layers.empty() && "sizes_of_layers empty");
    assert(sizes_of_layers.size() - names_of_activation_functions.size() == 1 &&
           "invalid parametrs");
    for (int i = 1; i < sizes_of_layers.size(); ++i) {
        AddLayer(In{sizes_of_layers[i - 1]}, Out{sizes_of_layers[i]},
                 names_of_activation_functions[i - 1]);
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

void Net::Backward(DataLoader& dl, LossFunc::Name name, int batch_size, int num_epochs,
                   double start_learning_rate, double beta1, double beta2) {
    assert(!layers_.empty() && "empty layers");
    std::vector<AdamOptimizer> optimizers;
    LossFunc loss_func(name);
    for (int i = 0; i < layers_.size(); ++i) {
        Index rows = layers_[i].GetWeightRows();
        Index cols = layers_[i].GetWeightCols();
        AdamOptimizer opt_of_weights(Rows{rows}, Cols{cols}, start_learning_rate, beta1, beta2);
        optimizers.emplace_back(Rows{rows}, Cols{cols}, start_learning_rate, beta1, beta2);
    }

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "EPOCH     " << epoch << std::endl;

        dl.ShuffleData();
        for (Data& train_batch : dl.Batches(batch_size)) {
            TrainBatch(train_batch, optimizers, loss_func);
        }
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
void Net::AddLayer(In input_size, Out output_size, ActivationFunc::Name name) {
    assert((layers_.empty() || layers_.back().GetWeightRows() == static_cast<Index>(input_size)) &&
           "incorrect layer");
    layers_.emplace_back(input_size, output_size, name);
}
}  // namespace network