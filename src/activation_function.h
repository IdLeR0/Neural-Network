#pragma once
#include <cmath>
#include <functional>
#include "global_usings.h"

namespace network {
class ActivationFunc {
    using Function = std::function<Vector(const Vector&)>;
    using Differential = std::function<Matrix(const Vector&)>;

public:
    ActivationFunc() = default;
    enum class Name { Sigmoid, ReLU, Tanh, Softmax, Id };
    explicit ActivationFunc(Name name);
    Matrix Apply(const Matrix& x) const;
    Matrix GetDifferential(const Vector& x) const;
    Index GetFuncId() const;
    bool operator==(const ActivationFunc& other) const;

private:
    Function apply_;
    Differential differential_;
    Index func_id_;
};

}  // namespace network
