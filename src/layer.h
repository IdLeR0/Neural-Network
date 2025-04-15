
#pragma once
#include "linalg.h"
#include "activation_function.h"
#include "file_reader_writer.h"
namespace network {

struct ParamsGrad {
    Matrix weights;
    Vector bias;
};

class RandomParams {
    using Generator = Eigen::Rand::Vmt19937_64;

public:
    RandomParams(int seed = kDefaultSeed);
    Matrix GenerateNormalMatrix(Index rows, Index cols, double mean = 0, double stdev = 1);
    Vector GenerateNormalVector(Index rows, double mean = 0.0, double stdev = 1.0);
    Matrix GenerateUniformMatrix(Index rows, Index cols, double low = 0.0, double high = 1.0);
    Vector GenerateUniformVector(Index rows, double low = 0.0, double high = 0.0);
    Matrix GenerateConstantMatrix(Index rows, Index cols, double value);
    Vector GenerateConstantVector(Index rows, double value);

private:
    static constexpr int kDefaultSeed = 42;
    Generator generator_{kDefaultSeed};
};

enum class In : Index;
enum class Out : Index;
class Layer {

public:
    Layer() = default;
    Layer(In input_size, Out output_size, ActivationFunc::Name name);

    Layer(const Matrix& weights, const Vector& bias, ActivationFunc::Name name);

    Matrix ApplyLinear(const Matrix& input) const;
    Matrix Forward(const Matrix& input) const;
    Matrix Backward(const Matrix& input_batch, const Matrix& gradient) const;
    ParamsGrad GetParametrsGradient(const Matrix& input_batch, const Matrix& gradient) const;
    void UpdateWeights(const Matrix& correction);
    void UpdateBias(const Vector& correction);

    friend FileWriter& operator<<(FileWriter& in, const Layer& layer);
    friend FileReader& operator>>(FileReader& out, Layer& layer);
    bool operator==(const Layer& other) const = default;
    Index GetWeightsRows() const;
    Index GetWeightsCols() const;

private:
    void InitializeParametrs(Index rows, Index cols, ActivationFunc::Name name);
    ActivationFunc func_;
    Matrix weights_;
    Vector bias_;
};

}  // namespace network