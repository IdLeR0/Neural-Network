
#pragma once
#include "global_usings.h"
#include "activation_function.h"
#include "file_reader_writer.h"
#include "random_params.h"

namespace network {

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
    LayerParams GetParametrsGradient(const Matrix& input_batch, const Matrix& gradient) const;
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