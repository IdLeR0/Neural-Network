#pragma once
#include "linalg.h"
#include <cmath>
namespace network {


enum class NamesLossFunctions { Mse, CrossEntorpy };

struct Mse {
  static double GetValue(const Vector &y_out, const Vector &y_expected);
  static Vector GetGradient(const Vector &y_out, const Vector &y_expected);
};
struct CrossEntropy {
  static double GetValue(const Vector &y_out, const Vector &y_expected);
  static Vector GetGradient(const Vector &y_out, const Vector &y_expected);
};

class LossFunc {
  public:
    LossFunc(NamesLossFunctions name);
    double Dist(const Vector &y_out, const Vector &y_expected);
    VectorT GetGradient(const Vector &y_out, const Vector &y_expected);
    Matrix GetGradient(const Matrix &y_out, const Matrix &y_expected);
  
  private:
    std::function<double(Vector, Vector)> loss_func_;
    std::function<Vector(Vector, Vector)> get_grad_;
  };
  
} // namespace Network