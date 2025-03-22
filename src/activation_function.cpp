
#include "activation_function.h"
#include <cassert>
namespace network {
namespace details {
struct Sigmoid {
    static Vector Apply(const Vector& x) {
        return x.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
    }
    static Matrix GetDifferential(const Vector& x) {
        Vector activated_vector = Apply(x);
        Vector diff = activated_vector.unaryExpr([](double x) { return (1.0 - x) * x; });
        return diff.asDiagonal();
    }
};

struct Tanh {
    static Vector Apply(const Vector& x) {
        return x.unaryExpr([](double x) { return std::tanh(x); });
    }
    static Matrix GetDifferential(const Vector& x) {
        Vector activated_vector = Apply(x);
        Vector diff = activated_vector.unaryExpr([](double x) { return 1.0 - x * x; });
        return diff.asDiagonal();
    }
};

struct ReLU {
    static Vector Apply(const Vector& x) {
        return x.unaryExpr([](double x) { return std::max(0.0, x); });
    }
    static Matrix GetDifferential(const Vector& x) {
        Vector diff = x.unaryExpr([](double x) { return x > 0.0 ? 1.0 : 0.0; });
        return diff.asDiagonal();
    }
};
struct Softmax {
    static Vector Apply(const Vector& x) {
        Vector tmp = x.unaryExpr([](double x) { return std::exp(x); });
        Vector activated_vector = tmp / tmp.sum();
        return activated_vector;
    }
    static Matrix GetDifferential(const Vector& x) {
        Vector tmp = Apply(x);
        Matrix jacobian(tmp.size(), tmp.size());
        for (int i = 0; i < tmp.size(); ++i) {
            for (int j = 0; j < tmp.size(); ++j) {
                jacobian(i, j) = tmp(i) * ((i == j) ? (1.0 - tmp(j)) : -tmp(j));
            }
        }
        return jacobian;
    }
};

}  // namespace details

void ActivationFunc::SetFunction(ActivationFunc::Name name) {
    switch (name) {
        case ActivationFunc::Name::Sigmoid:
            apply_ = details::Sigmoid::Apply;
            differential_ = details::Sigmoid::GetDifferential;
            break;
        case ActivationFunc::Name::ReLU:
            apply_ = details::ReLU::Apply;
            differential_ = details::ReLU::GetDifferential;
            break;
        case ActivationFunc::Name::Tanh:
            apply_ = details::Tanh::Apply;
            differential_ = details::Tanh::GetDifferential;
            break;
        case ActivationFunc::Name::Softmax:
            apply_ = details::Softmax::Apply;
            differential_ = details::Softmax::GetDifferential;
            break;
        default:
            assert(false && "invalid arguments in ActivationFunc constructor");
    }
}

ActivationFunc::ActivationFunc(ActivationFunc::Name name) {
    SetFunction(name);
}
Vector ActivationFunc::Apply(const Vector& vector) const {
    return apply_(vector);
}
Matrix ActivationFunc::Apply(const Matrix& x) const {
    Matrix res(x.rows(), x.cols());

    for (int i = 0; i < x.cols(); ++i) {
        res.col(i) = apply_(x.col(i));
    }

    return res;
}
Matrix ActivationFunc::GetDifferential(const Vector& x) const {
    return differential_(x);
}
}  // namespace network