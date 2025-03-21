#pragma once
#include "linalg.h"
#include <cmath>
namespace network {

class LossFunc {
public:
    enum class Name { Mse, CrossEntropy };
    LossFunc(Name name);
    double Dist(const Vector &y_out, const Vector &y_expected);
    VectorT GetGradient(const Vector &y_out, const Vector &y_expected);
    Matrix GetGradient(const Matrix &y_out, const Matrix &y_expected);

private:
    std::function<double(const Vector &, const Vector &)> loss_func_;
    std::function<Vector(const Vector &, const Vector &)> get_grad_;
};

}  // namespace network