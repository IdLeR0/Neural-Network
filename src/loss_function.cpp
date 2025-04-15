#include "loss_function.h"

namespace network {
namespace details {

struct Mse {
    struct Mae {
        static double GetValue(const Vector &y_out, const Vector &y_expected) {
            assert(y_out.size() == y_expected.size() && "Mse GetValue");
            return (y_out - y_expected).cwiseAbs().sum();
        }
        static Vector GetGradient(const Vector &y_out, const Vector &y_expected) {
            assert(y_out.size() == y_expected.size() && "Mse GetGradient");
            return (y_out - y_expected).unaryExpr([](double x) { return x > 0 ? 1.0 : -1.0; });
        }
    };

    static double GetValue(const Vector &y_out, const Vector &y_expected) {
        assert(y_out.size() == y_expected.size() && "Mse GetValue");
        return (y_out - y_expected).squaredNorm();
    }
    static Vector GetGradient(const Vector &y_out, const Vector &y_expected) {
        assert(y_out.size() == y_expected.size() && "Mse GetGradient");
        return 2.0 * (y_out - y_expected);
    }
};
struct CrossEntropy {
    static constexpr double kEPS = 1e-12;
    static double GetValue(const Vector &y_out, const Vector &y_expected) {
        assert(y_out.size() == y_expected.size() && "CrossEntropy GetValue");

        return -(y_expected.array() * (y_out.array() + kEPS).log()).sum();
    }
    static Vector GetGradient(const Vector &y_out, const Vector &y_expected) {
        assert(y_out.size() == y_expected.size() && "CrossEnropy GetGradient");

        return -(y_expected.array() / (y_out.array() + kEPS));
    }
};

}  // namespace details
LossFunc::LossFunc(Name name) {
    switch (name) {
        case Name::Mse:
            loss_func_ = details::Mse::GetValue;
            get_grad_ = details::Mse::GetGradient;
            break;
        case Name::CrossEntropy:
            loss_func_ = details::CrossEntropy::GetValue;
            get_grad_ = details::CrossEntropy::GetGradient;
            break;
        default:
            assert(false && "Problen in LossFunc constructor");
    }
}
double LossFunc::Dist(const Vector &y_out, const Vector &y_expected) {
    return loss_func_(y_out, y_expected);
}

Matrix LossFunc::GetGradient(const Matrix &y_out, const Matrix &y_expected) {
    assert(y_out.cols() == y_expected.cols() && y_out.rows() == y_expected.rows() &&
           "Matrices have different sizes. GetGradient.");
    Matrix res(y_out.cols(), y_out.rows());
    for (int i = 0; i < res.rows(); ++i) {
        res.row(i) = GetVectorGrad(y_out.col(i), y_expected.col(i));
    }
    return res;
}
VectorT LossFunc::GetVectorGrad(const Vector &y_out, const Vector &y_expected) {
    return get_grad_(y_out, y_expected).transpose();
}

}  // namespace network
