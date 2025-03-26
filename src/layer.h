
#pragma once
#include "linalg.h"
#include "activation_function.h"

namespace network {

struct WeightsBiasGradient {
    Matrix weights;
    Vector bias;
};
namespace details {

class Random {
    using Generator = Eigen::Rand::Vmt19937_64;

public:
    Random(int seed = kDefaultSeed);
    Matrix NormalMatrix(Index rows, Index cols, double mean = 0, double stdev = 1);
    Matrix ConstMatrix(Index rows, Index cols, double value);

private:
    static constexpr int kDefaultSeed = 42;
    Generator generator_{kDefaultSeed};
};
}  // namespace details

enum class In : Index;
enum class Out : Index;
class Layer {
    using Rand = details::Random;

public:
    Layer() = default;
    Layer(In input_size, Out output_size, ActivationFunc::Name name, Rand& rnd = GlobalRandom());
    // этот конструктор больше нужен для тестирование
    Layer(const Matrix& weights, const Vector& bias, ActivationFunc::Name name);

    Matrix ApplyLinear(const Matrix& input) const;
    Matrix Forward(const Matrix& input) const;
    Matrix Backward(const Matrix& input_batch, const Matrix& gradient) const;
    WeightsBiasGradient GetWeightsBiasGradient(const Matrix& input_batch,
                                               const Matrix& gradient) const;
    void UpdateWeights(const Matrix& correction);
    void UpdateBias(const Vector correction);
    Index GetWeightCols() const;
    Index GetWeightRows() const;
    // friend FileWriter& operator<<(FileWriter& in, const Layer& layer);
    // friend FileReader& operator>>(FileReader& out, Layer& layer);

private:
    static Rand& GlobalRandom();
    void InitializeParametrs(Index rows, Index cols, ActivationFunc::Name name, Rand& rnd);

    ActivationFunc func_;
    Matrix weights_;
    Vector bias_;
};

}  // namespace network