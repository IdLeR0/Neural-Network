#include "net.h"
#include "activation_function.h"

// НЕ ЧИТАЙ
namespace network {
Net::Net(DataLoader&& dl, LossFunc::Name name) : dl_(std::move(dl)), loss_func_(name) {
}
void Net::AddLayer(In input_size, Out output_size, ActivationFunc::Name name) {
    layers_.emplace_back(Layer(input_size, output_size, name));
}
std::vector<Matrix> Net::Forward(const Matrix& batch) {
    Matrix cur_output = batch;
    std::vector<Matrix> output_layers;
    output_layers.push_back(cur_output);
    for (const Layer& layer : layers_) {
        cur_output = layer.Forward(cur_output);
        output_layers.push_back(cur_output);
    }
    return output_layers;
}
void Backward() {
}
}  // namespace network