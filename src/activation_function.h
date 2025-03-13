#pragma once
#include "linalg.h"
#include <cmath>
#include <functional>
namespace network {
enum class NamesActivationFunc { Sigmoid, ReLU, Tanh, Softmax };

namespace details_activation_functions {
struct Sigmoid {
    static Vector Activate(const Vector& vector);
    static Matrix GetDifferential(const Vector& vector);
};

struct Tanh {
    static Vector Activate(const Vector& vector);
    static Matrix GetDifferential(const Vector& vector);
};

struct ReLU {
    static Vector Activate(const Vector& vector);
    static Matrix GetDifferential(const Vector& vector);
};
struct Softmax {
    static Vector Activate(const Vector& vector);
    static Matrix GetDifferential(const Vector& vector);
};
}  // namespace details_activation_functions

class ActivationFunc {
    using Function = std::function<Vector(Vector)>;
    using Differential = std::function<Matrix(Vector)>;
    // ActivationFunc(Function&& apply, Differential&& differential);

public:
    explicit ActivationFunc(NamesActivationFunc name);
    Vector Activate(const Vector& vector);
    Matrix GetDifferential(const Vector& vector);

private:
    Function apply_;
    Differential differential_;
};

}  // namespace network
