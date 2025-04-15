
#include "activation_function.h"
#include <cassert>
#include "file_reader_writer.h"
namespace network {
namespace details {
struct Sigmoid {
    static Vector Apply(const Vector& x) {
        return x.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
    }
    static Matrix GetDifferential(const Vector& x) {
        Vector applied_sigmoid = Apply(x);
        Vector differential = applied_sigmoid.unaryExpr([](double x) { return (1.0 - x) * x; });
        return differential.asDiagonal();
    }
};

struct Tanh {
    static Vector Apply(const Vector& x) {
        return x.unaryExpr([](double x) { return std::tanh(x); });
    }
    static Matrix GetDifferential(const Vector& x) {
        Vector applied_tanh = Apply(x);
        Vector differential = applied_tanh.unaryExpr([](double x) { return 1.0 - x * x; });
        return differential.asDiagonal();
    }
};

struct ReLU {
    static Vector Apply(const Vector& x) {
        return x.unaryExpr([](double x) { return std::max(0.0, x); });
    }
    static Matrix GetDifferential(const Vector& x) {
        Vector differential = x.unaryExpr([](double x) { return x > 0.0 ? 1.0 : 0.0; });
        return differential.asDiagonal();
    }
};
struct Softmax {
    static Vector Apply(const Vector& x) {
        Vector tmp = x.unaryExpr([](double x) { return std::exp(x); });
        Vector applied_softmax = tmp / tmp.sum();
        return applied_softmax;
    }
    static Matrix GetDifferential(const Vector& x) {
        Vector applied_softmax = Apply(x);
        Matrix tmp = applied_softmax.asDiagonal();
        return tmp - applied_softmax * applied_softmax.transpose();
    }
};
struct Id {
    static Vector Apply(const Vector& x) {
        return x;
    }
    static Matrix GetDifferential(const Vector& x) {
        return Eigen::MatrixXd::Identity(x.rows(), x.rows());
    }
};

}  // namespace details

ActivationFunc::ActivationFunc(ActivationFunc::Name name) {
    switch (name) {
        case ActivationFunc::Name::Sigmoid:
            func_id_ = static_cast<int>(ActivationFunc::Name::Sigmoid);
            apply_ = details::Sigmoid::Apply;
            differential_ = details::Sigmoid::GetDifferential;
            break;
        case ActivationFunc::Name::ReLU:
            func_id_ = static_cast<int>(ActivationFunc::Name::ReLU);
            apply_ = details::ReLU::Apply;
            differential_ = details::ReLU::GetDifferential;
            break;
        case ActivationFunc::Name::Tanh:
            func_id_ = static_cast<int>(ActivationFunc::Name::Tanh);
            apply_ = details::Tanh::Apply;
            differential_ = details::Tanh::GetDifferential;
            break;
        case ActivationFunc::Name::Softmax:
            func_id_ = static_cast<int>(ActivationFunc::Name::Softmax);
            apply_ = details::Softmax::Apply;
            differential_ = details::Softmax::GetDifferential;
            break;
        case ActivationFunc::Name::Id:
            func_id_ = static_cast<int>(ActivationFunc::Name::Id);
            apply_ = details::Id::Apply;
            differential_ = details::Id::GetDifferential;
            break;

        default:
            assert(false && "invalid arguments in ActivationFunc constructor");
    }
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
int ActivationFunc::GetFuncId() const {
    return func_id_;
}
bool ActivationFunc::operator==(const ActivationFunc& other) const {
    return GetFuncId() == other.GetFuncId();
}

}  // namespace network