#pragma once
#include <cmath>
#include <functional>
#include "linalg.h"

namespace network {
class ActivationFunc {
    using Function = std::function<Vector(const Vector&)>;
    using Differential = std::function<Matrix(const Vector&)>;

public:
    ActivationFunc() = default;
    enum class Name { Sigmoid, ReLU, Tanh, Softmax, Linear };
    explicit ActivationFunc(Name name);
    void SetFunction(Name name);
    Matrix Apply(const Matrix& x) const;
    Matrix GetDifferential(const Vector& x) const;
    int GetId() const;

private:
    Function apply_;
    Differential differential_;
    int id_;
};

}  // namespace network
