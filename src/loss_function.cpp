#include "loss_function.h"

namespace network {
    
double Mse::GetValue(const Vector &y_out, const Vector &y_expected) {
  assert(y_out.size() == y_expected.size() && "Mse GetValue");
  return (y_out - y_expected).array().square().mean();
}
Vector Mse::GetGradient(const Vector &y_out, const Vector &y_expected) {

  assert(y_out.size() == y_expected.size() && "Mse GetGradient");
  return (2.0 / y_out.size()) * (y_out - y_expected);
}
double CrossEntropy::GetValue(const Vector &y_out, const Vector &y_expected) {

  assert(y_out.size() == y_expected.size() && "CrossEntropy GetValue");
  const double eps = 1e-12;
  return -(y_expected.array() * (y_out.array() + eps).log()).sum();
}
Vector CrossEntropy::GetGradient(const Vector &y_out,
                                 const Vector &y_expected) {

  assert(y_out.size() == y_expected.size() && "CrossEnropy GetGradient");
  const double eps = 1e-12;
  return -(y_expected.array() / (y_out.array() + eps));
}
LossFunc::LossFunc(NamesLossFunctions name) {
  switch (name) {
  case NamesLossFunctions::Mse:
    loss_func_ = Mse::GetValue;
    get_grad_ = Mse::GetGradient;
  case network::NamesLossFunctions::CrossEntorpy:
    loss_func_ = CrossEntropy::GetValue;
    get_grad_ = CrossEntropy::GetGradient;
  default:
    assert("Problen in LossFunc constructor");
  }
}
double LossFunc::Dist(const Vector &y_out, const Vector &y_expected) {
  return loss_func_(y_out, y_expected);
}
VectorT LossFunc::GetGradient(const Vector &y_out, const Vector &y_expected) {
  return get_grad_(y_out, y_expected).transpose();
}
Matrix GetGradient(const Matrix &y_out, const Matrix &y_expected) {
  assert(y_out.cols() == y_expected.cols() &&
         y_out.rows() == y_expected.rows() &&
         "Matrices have different sizes. GetGradient.");
  Matrix res(y_out.rows(), y_out.cols());
  for (int i = 0; i < y_out.cols(); ++i) {
    VectorT grad_of_one_vector = 
    res.row(i) = GetGradient(y_out.col(i), y_expected.col(i));
  }
  return res;
}

} // namespace Network
