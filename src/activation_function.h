#pragma once
#include "linalg.h"
#include <cmath>
#include <functional>
namespace network {
class ActivationFunc {
    using Function = std::function<Vector(const Vector&)>;
    using Differential = std::function<Matrix(const Vector&)>;

public:
    enum class Name { Sigmoid, ReLU, Tanh, Softmax };
    explicit ActivationFunc(Name name);
    void SetFunction(Name name);
    Vector Apply(const Vector& x) const;
    Matrix Apply(const Matrix& x) const;
    Matrix GetDifferential(const Vector& x) const;

private:
    Function apply_;
    Differential differential_;
};

}  // namespace network
