#include "layer.h"
#include "activation_function.h"
#include <cassert>
namespace network {
namespace details {

Random::Random(int seed) : generator_(seed) {
}
// кажется ты говорил что лучше заполнять нормальным распределением, а не равномерным
Matrix Random::NormalMatrix(Index rows, Index cols, double mean, double stdev) {
    return Eigen::Rand::normal<Matrix>(rows, cols, generator_, mean, stdev);
}
Vector Random::NormalVector(Index rows, double mean, double stdev) {
    return Eigen::Rand::normal<Vector>(rows, 1, generator_, mean, stdev);
}

}  // namespace details

Layer::Layer(In input_size, Out output_size, ActivationFunc::Name name, Rand& rnd)
    : weights_(
          InitializedWeights(static_cast<Index>(output_size), static_cast<Index>(input_size), rnd)),
      bias_(InitializedBias(static_cast<Index>(output_size), rnd)),
      func_(name) {
}
Layer::Layer(const Matrix& weights, const Vector& bias, ActivationFunc::Name name)
    : weights_(weights), bias_(bias), func_(name) {
}
Matrix Layer::ApplyLinear(const Matrix& input_batch) const {
    assert(weights_.cols() == input_batch.rows() &&
           "can not multiply input_batch on weights matrix");
    Matrix output_batch = weights_ * input_batch;
    output_batch.colwise() += bias_;
    return output_batch;
}

// Vector Layer::Forward(const Vector& input) {
//     assert(input.size() == weights_.cols() && "wrong size of layer");
//     Vector output = weights_ * input + bias_;
//     output = func_.Activate(output);
//     return output;
// }
Matrix Layer::Forward(const Matrix& input) const {
    return func_.Apply(ApplyLinear(input));
}

void Layer::UpdateWeights(const Matrix& correction) {
    assert(weights_.cols() == correction.cols() && weights_.rows() == correction.rows() &&
           "invalid correction");
    weights_ -= correction;
}
void Layer::UpdateBias(const Vector correction) {
    assert(correction.rows() == bias_.rows() && "invalid correction");
    bias_ -= correction;
}

Matrix Layer::Backward(const Matrix& input_batch, const Matrix& gradient) const {
    assert(input_batch.cols() == gradient.rows() && "different size of gradient and input_batch");
    assert(weights_.rows() == gradient.cols() && "wrong size of gradient or of weights matrix");
    assert(weights_.cols() == input_batch.rows() && "wrong size of weight matrix or input_batch");
    Matrix applied_linear = ApplyLinear(input_batch);
    Matrix tmp = gradient;
    for (int i = 0; i < tmp.rows(); ++i) {
        Matrix act_func_differ = func_.GetDifferential(applied_linear.col(i));
        tmp.row(i) = tmp.row(i) * act_func_differ;
    }
    Matrix new_gradient(input_batch.cols(), input_batch.rows());
    new_gradient = tmp * weights_;
    return new_gradient;
}

WeightsBiasGradient Layer::GetWeightsBiasGradient(const Matrix& input_batch,
                                                  const Matrix& gradient) const {
    assert(input_batch.cols() == gradient.rows() && "different size of gradient and input_batch");
    assert(weights_.rows() == gradient.cols() && "wrong size of gradient or of weights matrix");
    assert(weights_.cols() == input_batch.rows() && "wrong size of weight matrix or input_batch");
    WeightsBiasGradient grad;
    Matrix applied_linear = ApplyLinear(input_batch);
    Matrix matrix_grad_biases(bias_.rows(), input_batch.cols());
    for (int i = 0; i < input_batch.cols(); ++i) {
        matrix_grad_biases.col(i) =
            (gradient.row(i) * func_.GetDifferential(applied_linear.col(i))).transpose();
    }
    grad.bias = matrix_grad_biases.rowwise().sum() / matrix_grad_biases.cols();
    grad.weights = matrix_grad_biases * input_batch.transpose() / matrix_grad_biases.cols();
    return grad;
}
Index Layer::GetWeightCols() const {
    return weights_.cols();
}
Index Layer::GetWeightRows() const {
    return weights_.rows();
}

Matrix Layer::InitializedWeights(Index rows, Index cols, Rand& rnd) {
    return rnd.NormalMatrix(rows, cols);
}
Vector Layer::InitializedBias(Index rows, Rand& rnd) {
    return rnd.NormalVector(rows);
}
Layer::Rand& Layer::GlobalRandom() {
    static Rand rnd;
    return rnd;
}
}  // namespace network