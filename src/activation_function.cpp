
#include "activation_function.h"
#include <cassert>
namespace network {
namespace details_activation_functions {
Vector Sigmoid::Activate(const Vector &vector) {
    return vector.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
}

Matrix Sigmoid::GetDifferential(const Vector &vector) {
    Vector activated_vector = Activate(vector);
    Vector diff = activated_vector.unaryExpr([](double x) { return (1.0 - x) * x; });
    return diff.asDiagonal();
}

Vector Tanh::Activate(const Vector &vector) {
    return vector.unaryExpr([](double x) { return std::tanh(x); });
}

Matrix Tanh::GetDifferential(const Vector &vector) {
    Vector activated_vector = Activate(vector);
    Vector diff = activated_vector.unaryExpr([](double x) { return 1.0 - x * x; });
    return diff.asDiagonal();
}
Vector ReLU::Activate(const Vector &vector) {
    return vector.unaryExpr([](double x) { return std::max(0.0, x); });
}

Matrix ReLU::GetDifferential(const Vector &vector) {
    Vector diff = vector.unaryExpr([](double x) { return x > 0.0 ? 1.0 : 0.0; });
    return diff.asDiagonal();
}
Vector Softmax::Activate(const Vector &vector) {
    Vector tmp = vector.unaryExpr([](double x) { return std::exp(x); });
    Vector activated_vector = tmp / tmp.sum();
    return activated_vector;
}
Matrix Softmax::GetDifferential(const Vector &vector) {
    Vector tmp = Activate(vector);
    Matrix jacobian(tmp.size(), tmp.size());
    for (int i = 0; i < tmp.size(); ++i) {
        for (int j = 0; j < tmp.size(); ++j) {
            jacobian(i, j) = tmp(i) * ((i == j) ? (1.0 - tmp(j)) : -tmp(j));
        }
    }
    return jacobian;
}

}  // namespace details_activation_functions

ActivationFunc::ActivationFunc(NamesActivationFunc name) {
    switch (name) {
        case NamesActivationFunc::Sigmoid:
            apply_ = details_activation_functions::Sigmoid::Activate;
            differential_ = details_activation_functions::Sigmoid::GetDifferential;
            break;
        case NamesActivationFunc::ReLU:
            apply_ = details_activation_functions::ReLU::Activate;
            differential_ = details_activation_functions::ReLU::GetDifferential;
            break;
        case NamesActivationFunc::Tanh:
            apply_ = details_activation_functions::Tanh::Activate;
            differential_ = details_activation_functions::Tanh::GetDifferential;
            break;
        case NamesActivationFunc::Softmax:
            apply_ = details_activation_functions::Softmax::Activate;
            differential_ = details_activation_functions::Softmax::GetDifferential;
        default:
            assert("invalid arguments in ActivationFunc constructor");
    }
}
Vector ActivationFunc::Activate(const Vector &vector) {
    return apply_(vector);
}
Matrix ActivationFunc::GetDifferential(const Vector &vector) {
    return differential_(vector);
}
}  // namespace network