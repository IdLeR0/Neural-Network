#include "net.h"
#include <codecvt>
#include "activation_function.h"
#include "adam_optimizer.h"
#include "layer.h"
#include <iostream>
namespace network {
Net::Net(DataLoader&& dl, LossFunc::Name name) : dl_(std::move(dl)), loss_func_(name) {
}
void Net::AddLayer(In input_size, Out output_size, ActivationFunc::Name name) {
    assert((layers_.empty() || layers_.back().GetWeightRows() == static_cast<Index>(input_size)) &&
           "incorrect layer");
    layers_.emplace_back(In{input_size}, Out{output_size}, name);
}
Net::ComputedBatches Net::Forward(const Matrix& batch) const {
    Matrix cur_input = batch;
    std::vector<Matrix> inputs;
    inputs.push_back(cur_input);
    for (const Layer& layer : layers_) {
        cur_input = layer.Forward(cur_input);
        inputs.push_back(cur_input);
    }
    return inputs;
}

Matrix Net::Evalute(const Matrix& batch) const {
    Matrix cur_batch = batch;
    for (const Layer& layer : layers_) {
        cur_batch = layer.Forward(cur_batch);
    }
    return cur_batch;
}

void Net::Backward(int batch_size, int num_epochs, double start_learning_rate, double beta1,
                   double beta2) {
    assert(!layers_.empty() && "empty layers");
    for (int i = 0; i < layers_.size(); ++i) {

        Index rows = layers_[i].GetWeightRows();
        Index cols = layers_[i].GetWeightCols();
        AdamOptimizer opt_of_weights(Rows{rows}, Cols{cols}, start_learning_rate, beta1, beta2);
        AdamOptimizer opt_of_bias(Rows{rows}, Cols{1}, start_learning_rate, beta1, beta2);
        opts_.push_back({opt_of_weights, opt_of_bias});
    }
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        dl_.ShuffleData();
        for (const Data& train_batch : dl_.Batches(batch_size)) {
            TrainBatch(train_batch);
        }
    }
}

void Net::TrainBatch(const Data& data) {
    assert(!layers_.empty() && "no layers");
    assert(layers_.size() == opts_.size() && "different sizes of optimizers and layers");
    ComputedBatches computed_batces = Forward(data.input);
    assert(computed_batces.size() - layers_.size() == 1 &&
           "invalide size of layers_ or computed_batches");
    Matrix cur_gradient = loss_func_.GetGradient(computed_batces.back(), data.output);
    for (int i = layers_.size() - 1; i >= 0; --i) {
        WeightsBiasGradient grad =
            layers_[i].GetWeightsBiasGradient(computed_batces[i], cur_gradient);
        Matrix correction_weights = opts_[i].weights.ComputeCorrection(grad.weights);
        Vector correction_bias = opts_[i].bias.ComputeCorrection(grad.bias);
        cur_gradient = layers_[i].Backward(computed_batces[i], cur_gradient);
        layers_[i].UpdateWeights(correction_weights);
        layers_[i].UpdateBias(correction_bias);
    }
}

}  // namespace network