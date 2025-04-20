#include "layer.h"
#include "activation_function.h"
#include "file_reader_writer.h"
#include <cassert>
#include <cmath>

namespace network {
namespace {
RandomParams& GetRandomGenerator() {
    static RandomParams rnd;
    return rnd;
}
}  // namespace

Layer::Layer(In input_size, Out output_size, ActivationFunc::Name name) : func_(name) {
    InitializeParametrs(static_cast<Index>(output_size), static_cast<Index>(input_size), name);
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

Matrix Layer::Forward(const Matrix& input) const {
    return func_.Apply(ApplyLinear(input));
}

void Layer::UpdateWeights(const Matrix& correction) {
    assert(weights_.cols() == correction.cols() && weights_.rows() == correction.rows() &&
           "invalid correction");
    weights_ -= correction;
}

void Layer::UpdateBias(const Vector& correction) {
    assert(correction.rows() == bias_.rows() && "invalid correction");
    bias_ -= correction;
}

Matrix Layer::Backward(const Matrix& input_batch, const Matrix& gradient) const {
    assert(input_batch.cols() == gradient.rows() && "different size of gradient and input_batch");
    assert(weights_.rows() == gradient.cols() && "wrong size of gradient or of weights matrix");
    assert(weights_.cols() == input_batch.rows() && "wrong size of weight matrix or input_batch");
    Matrix applied_linear = ApplyLinear(input_batch);
    Matrix tmp = gradient;
    for (Index i = 0; i < tmp.rows(); ++i) {
        Matrix act_func_differ = func_.GetDifferential(applied_linear.col(i));
        tmp.row(i) = tmp.row(i) * act_func_differ;
    }
    Matrix new_gradient(input_batch.cols(), input_batch.rows());
    new_gradient = tmp * weights_;
    return new_gradient;
}

LayerParams Layer::GetParametrsGradient(const Matrix& input_batch, const Matrix& gradient) const {
    assert(input_batch.cols() == gradient.rows() && "different size of gradient and input_batch");
    assert(weights_.rows() == gradient.cols() && "wrong size of gradient or of weights matrix");
    assert(weights_.cols() == input_batch.rows() && "wrong size of weight matrix or input_batch");
    LayerParams grad;
    Matrix applied_linear = ApplyLinear(input_batch);
    Matrix matrix_grad_biases(bias_.rows(), input_batch.cols());
    for (Index i = 0; i < input_batch.cols(); ++i) {
        matrix_grad_biases.col(i) =
            (gradient.row(i) * func_.GetDifferential(applied_linear.col(i))).transpose();
    }
    grad.bias = matrix_grad_biases.rowwise().sum() / matrix_grad_biases.cols();
    grad.weights = matrix_grad_biases * input_batch.transpose() / matrix_grad_biases.cols();
    return grad;
}

FileWriter& operator<<(FileWriter& in, const Layer& layer) {
    in << layer.func_.GetFuncId();
    in << layer.weights_;
    in << layer.bias_;
    return in;
}

FileReader& operator>>(FileReader& out, Layer& layer) {
    Index id;
    Matrix weights;
    Vector bias;
    out >> id;
    out >> weights;
    out >> bias;
    layer = Layer(weights, bias, static_cast<ActivationFunc::Name>(id));
    return out;
}

Index Layer::GetWeightsRows() const {
    return weights_.rows();
}

Index Layer::GetWeightsCols() const {
    return weights_.cols();
}

void Layer::InitializeParametrs(Index rows, Index cols, ActivationFunc::Name name) {
    RandomParams rnd = GetRandomGenerator();
    constexpr DataType kConst = 0.01;
    DataType stdev;
    switch (name) {
        case ActivationFunc::Name::ReLU:
            stdev = std::sqrt(2.0 / static_cast<DataType>(cols));
            weights_ = rnd.GenerateNormalMatrix(rows, cols, 0, stdev);
            bias_ = rnd.GenerateConstantVector(rows, kConst);
            break;

        case ActivationFunc::Name::Id:
            weights_ = rnd.GenerateNormalMatrix(rows, cols, 0, kConst);
            bias_ = rnd.GenerateConstantVector(rows, 0);
            break;

        default:
            stdev = std::sqrt(2.0 / (static_cast<DataType>(cols) + static_cast<DataType>(rows)));
            weights_ = rnd.GenerateNormalMatrix(rows, cols, 0, stdev);
            bias_ = rnd.GenerateConstantVector(rows, 0);
            break;
    }
}

}  // namespace network