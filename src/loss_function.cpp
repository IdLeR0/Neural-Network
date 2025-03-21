#include "loss_function.h"

namespace network {
namespace details {

struct Mse {

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
        case Name::CrossEntropy:
            loss_func_ = details::CrossEntropy::GetValue;
            get_grad_ = details::CrossEntropy::GetGradient;
        default:
            assert(false && "Problen in LossFunc constructor");
    }
}
double LossFunc::Dist(const Vector &y_out, const Vector &y_expected) {
    return loss_func_(y_out, y_expected);
}
VectorT LossFunc::GetGradient(const Vector &y_out, const Vector &y_expected) {
    return get_grad_(y_out, y_expected).transpose();
}
Matrix GetGradient(const Matrix &y_out, const Matrix &y_expected) {
    assert(y_out.cols() == y_expected.cols() && y_out.rows() == y_expected.rows() &&
           "Matrices have different sizes. GetGradient.");
    Matrix res(y_out.rows(), y_out.cols());
    for (int i = 0; i < y_out.cols(); ++i) {
        res.row(i) = GetGradient(y_out.col(i), y_expected.col(i));
    }
    return res;
}

}  // namespace network
