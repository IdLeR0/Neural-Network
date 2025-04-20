#pragma once
#include "global_usings.h"
#include <cmath>

namespace network {

class LossFunc {
    using Function = std::function<DataType(const Vector &, const Vector &)>;
    using Differential = std::function<Vector(const Vector &, const Vector &)>;

public:
    enum class Name { Mae, Mse, CrossEntropy };
    LossFunc(Name name);
    DataType Apply(const Vector &y_out, const Vector &y_expected) const;
    Matrix GetGradient(const Matrix &y_out, const Matrix &y_expected) const;

private:
    VectorT GetVectorGrad(const Vector &y_out, const Vector &y_expected) const;
    Function loss_func_;
    Differential get_grad_;
};

}  // namespace network