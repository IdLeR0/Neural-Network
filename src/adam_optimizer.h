#pragma once
#include "linalg.h"

namespace network {
enum class Rows : Index;
enum class Cols : Index;

class AdamOptimizer {
    struct Moments {
        Matrix weights_m_t;
        Matrix weights_v_t;
        Vector bias_m_t;
        Vector bias_v_t;
    };

public:
    AdamOptimizer(Rows rows, Cols cols, double start_learning_rate = kDefaultStartLearningRate,
                  double beta1 = kDefaultBeta1, double beta2 = kDefaultBeta2);
    void GetCorrection(Matrix& weights_gradient, Vector& bias_gradient);

private:
    static constexpr double kDefaultStartLearningRate = 3e-4;
    static constexpr double kDefaultBeta1 = 0.9;
    static constexpr double kDefaultBeta2 = 0.99;
    static constexpr double kEps = 1e-8;
    int t_;
    double learning_rate_;
    double beta1_;
    double beta2_;
    Moments moments_;
};
}  // namespace network