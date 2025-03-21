#include "adam_optimizer.h"
#include <cmath>

namespace network {

AdamOptimizer::AdamOptimizer(const Matrix& parametr, double start_learning_rate, double beta1,
                             double beta2)
    : t_(0),
      learning_rate_(start_learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      m_t_(Matrix::Zero(parametr.rows(), parametr.cols())),
      v_t_(Matrix::Zero(parametr.rows(), parametr.cols())) {
}

Matrix AdamOptimizer::ComputeUpdate(const Matrix& gradient) {
    assert(gradient.rows() == m_t_.rows() && gradient.cols() && m_t_.cols() && "invalid gradient");
    t_ += 1;
    m_t_ = beta1_ * m_t_ + (1 - beta1_) * gradient;
    Matrix squared_gradient = gradient.cwiseProduct(gradient);
    v_t_ = beta2_ * v_t_ + (1 - beta2_) * squared_gradient;
    double m_bias_correction = 1.0 - std::pow(beta1_, t_);
    double v_bias_correction = 1.0 - std::pow(beta2_, t_);
    Matrix m_tmp = m_t_ / m_bias_correction;
    Matrix v_tmp = v_t_ / v_bias_correction;
    return learning_rate_ * m_tmp.array() / (v_tmp.array().sqrt() + kEps);
}

}  // namespace network
