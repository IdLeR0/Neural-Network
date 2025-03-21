#pragma once
#include "linalg.h"

namespace network {
class AdamOptimizer {
public:
    AdamOptimizer(const Matrix& parametr, double start_learning_rate = kDefaultStartLearningRate,
                  double beta1 = kDefaultBeta1, double beta2 = kDefaultBeta2);
    Matrix ComputeUpdate(const Matrix& gradient);

private:
    static constexpr double kDefaultStartLearningRate = 3e-4;
    static constexpr double kDefaultBeta1 = 0.9;
    static constexpr double kDefaultBeta2 = 0.99;
    static constexpr double kEps = 1e-8;
    int t_;
    double learning_rate_;
    double beta1_;
    double beta2_;
    Matrix m_t_;
    Matrix v_t_;
};
}  // namespace network