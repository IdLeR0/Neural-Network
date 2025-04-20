#pragma once
#include "global_usings.h"

namespace network {
enum class Rows : Index;
enum class Cols : Index;

class AdamOptimizer {
public:
    AdamOptimizer(Rows rows, Cols cols, DataType learning_rate = kDefaultStartLearningRate,
                  DataType beta1 = kDefaultBeta1, DataType beta2 = kDefaultBeta2);
    LayerParams GetCorrection(const LayerParams& gradient);

private:
    struct Moments {
        Matrix weights_m_t;
        Matrix weights_v_t;
        Vector bias_m_t;
        Vector bias_v_t;
    };

    static constexpr DataType kDefaultStartLearningRate = 3e-4;
    static constexpr DataType kDefaultBeta1 = 0.9;
    static constexpr DataType kDefaultBeta2 = 0.99;
    static constexpr DataType kEps = 1e-8;
    Index t_;
    DataType learning_rate_;
    DataType beta1_;
    DataType beta2_;
    Moments moments_;
};
}  // namespace network