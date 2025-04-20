#include "adam_optimizer.h"
#include <cmath>

namespace network {

AdamOptimizer::AdamOptimizer(Rows rows, Cols cols, DataType learning_rate, DataType beta1,
                             DataType beta2)
    : t_(0),
      learning_rate_(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      moments_{Matrix::Zero(static_cast<Index>(rows), static_cast<Index>(cols)),
               Matrix::Zero(static_cast<Index>(rows), static_cast<Index>(cols)),
               Vector::Zero(static_cast<Index>(rows)), Vector::Zero(static_cast<Index>(rows))} {
}

LayerParams AdamOptimizer::GetCorrection(const LayerParams& gradient) {
    assert(gradient.weights.rows() == moments_.weights_m_t.rows() && "invalid weights_gradient");
    assert(gradient.weights.cols() == moments_.weights_m_t.cols() && "invalid weights_gradient");
    assert(gradient.bias.size() == moments_.bias_m_t.size() && "invalid bias_gradient");
    t_ += 1;
    moments_.weights_m_t = beta1_ * moments_.weights_m_t + (1 - beta1_) * gradient.weights;
    moments_.bias_m_t = beta1_ * moments_.bias_m_t + (1 - beta1_) * gradient.bias;

    Matrix squared_weights_gradient = gradient.weights.cwiseProduct(gradient.weights);
    Vector squared_bias_gradient = gradient.bias.cwiseProduct(gradient.bias);
    moments_.weights_v_t = beta2_ * moments_.weights_v_t + (1 - beta2_) * squared_weights_gradient;
    moments_.bias_v_t = beta2_ * moments_.bias_v_t + (1 - beta2_) * squared_bias_gradient;
    DataType m_correction = 1.0 - std::pow(beta1_, t_);
    DataType v_correction = 1.0 - std::pow(beta2_, t_);
    Matrix weights_correction = learning_rate_ * (moments_.weights_m_t / m_correction).array() /
                                ((moments_.weights_v_t / v_correction).array().sqrt() + kEps);
    Matrix bias_correction = learning_rate_ * (moments_.bias_m_t / m_correction).array() /
                             ((moments_.bias_v_t / v_correction).array().sqrt() + kEps);
    LayerParams correction{weights_correction, bias_correction};
    return correction;
}

}  // namespace network
