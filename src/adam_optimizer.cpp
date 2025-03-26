#include "adam_optimizer.h"
#include <cmath>

namespace network {

AdamOptimizer::AdamOptimizer(Rows rows, Cols cols, double start_learning_rate, double beta1,
                             double beta2)
    : t_(0),
      learning_rate_(start_learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      moments_{Matrix::Zero(static_cast<Index>(rows), static_cast<Index>(cols)),
               Matrix::Zero(static_cast<Index>(rows), static_cast<Index>(cols)),
               Vector::Zero(static_cast<Index>(rows)), Vector::Zero(static_cast<Index>(rows))} {
}

void AdamOptimizer::GetCorrection(Matrix& weights_gradient, Vector& bias_gradient) {
    assert(weights_gradient.rows() == moments_.weights_m_t.rows() && "invalid weights_gradient");
    assert(weights_gradient.cols() == moments_.weights_m_t.cols() && "invalid weights_gradient");
    assert(bias_gradient.size() == moments_.bias_m_t.size() && "invalid bias_gradient");
    t_ += 1;
    moments_.weights_m_t = beta1_ * moments_.weights_m_t + (1 - beta1_) * weights_gradient;
    moments_.bias_m_t = beta1_ * moments_.bias_m_t + (1 - beta1_) * bias_gradient;

    Matrix squared_weights_gradient = weights_gradient.cwiseProduct(weights_gradient);
    Vector squared_bias_gradient = bias_gradient.cwiseProduct(bias_gradient);
    moments_.weights_v_t = beta2_ * moments_.weights_v_t + (1 - beta2_) * squared_weights_gradient;
    moments_.bias_v_t = beta2_ * moments_.bias_v_t + (1 - beta2_) * squared_bias_gradient;
    double m_correction = 1.0 - std::pow(beta1_, t_);
    double v_correction = 1.0 - std::pow(beta2_, t_);
    weights_gradient = learning_rate_ * (moments_.weights_m_t / m_correction).array() /
                       ((moments_.weights_v_t / v_correction).array().sqrt() + kEps);
    bias_gradient = learning_rate_ * (moments_.bias_m_t / m_correction).array() /
                    ((moments_.bias_v_t / v_correction).array().sqrt() + kEps);
}

}  // namespace network
