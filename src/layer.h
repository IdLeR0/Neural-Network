
#pragma once
#include "activation_function.h"

#include "linalg.h"

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
    // лучше нормальное или равномерное?
    Matrix NormalMatrix(Index rows, Index cols, double mean = 0, double stdev = 1);
    Vector NormalVector(Index rows, double mean = 0, double stdev = 1);

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
    Layer(In input_size, Out output_size, ActivationFunc::Name name, Rand& rnd = GlobalRandom());
    // этот конструктор больше нужен для тестирование
    Layer(const Matrix& weights, const Vector& bias, ActivationFunc::Name name);

    Matrix ApplyLinear(const Matrix& input) const;
    // Vector Forward(const Vector& input);
    Matrix Forward(const Matrix& input) const;
    // Vector Backward(const Vector& gradient);
    Matrix Backward(const Matrix& input_batch, const Matrix& gradient) const;

    // наверно это плохая идея, но я думал так, чтобы не пересчитывать матрицу градиентов по b; Эта
    // структора сделана лишь с этой целью.
    WeightsBiasGradient GetWeightsBiasGradient(const Matrix& input_batch,
                                               const Matrix& gradient) const;
    void UpdateWeights(const Matrix& correction);
    void UpdateBias(const Vector correction);
    Index GetWeightCols() const;
    Index GetWeightRows() const;

private:
    static Rand& GlobalRandom();
    Matrix InitializedWeights(Index rows, Index cols, Rand& rnd);
    Vector InitializedBias(Index rows, Rand& rnd);
    ActivationFunc func_;
    Matrix weights_;
    Vector bias_;
};

}  // namespace network