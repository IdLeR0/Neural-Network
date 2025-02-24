#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <functional>
namespace Network {
enum class NameActivationFunc { Sigmoid, ReLU, Tanh };
double Sigmoid(double x);
double SigmoidDerivative(double x);

double Tanh(double x);
double TanhDerivative(double x);

double ReLU(double x);
double ReLUDerivative(double x);
// using в своем namespace
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
class ActivationFunc {
public:
  ActivationFunc() = default;

  // Я вспомнил пример про размеры картинок и не захотелась передавать 2
  // std::function. С другой стороны тут функции активации, котороые мы можем передовать строго фиксированы и пользователь не сможет передать свою функцию. 

  ActivationFunc(NameActivationFunc name);
  Vector Activate(Vector vector);
  Matrix GetDifferential(Vector vector);

private:
  // или лучше функтор/стирающиеся типы?
  std::function<double(double)> func_;
  std::function<double(double)> derivative_;
};

} 
