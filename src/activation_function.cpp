#include "activation_function.h"
#include <stdexcept>
namespace Network{
  double Sigmoid(double x) { return 1 / (1 + std::exp(-x)); }

double SigmoidDerivative(double x) {
  double sigmoid = Sigmoid(x);
  return (1 - sigmoid) * sigmoid;
}

double Tanh(double x) { return std::tanh(x); }

double TanhDerivative(double x) {
  double tanh_x = std::tanh(x);
  return 1 - tanh_x * tanh_x;
}

double ReLU(double x) { return std::max(x, 0.0); }

double ReLUDerivative(double x) { return x > 0 ? 1 : 0; }

ActivationFunc::ActivationFunc(NameActivationFunc name) {
  switch (name) {
  case NameActivationFunc::Sigmoid:
    func_ = Sigmoid;
    derivative_ = SigmoidDerivative;
    break;
  case NameActivationFunc::ReLU:
    func_ = ReLU;
    derivative_ = ReLUDerivative;
    break;
  case NameActivationFunc::Tanh:
    func_ = Tanh;
    derivative_ = TanhDerivative;
    break;
  default:
    std::invalid_argument("Trouble in constructor of Activationfunc");
  }
}
Vector ActivationFunc::Activate(Vector vector){
    Vector activated_vector(vector.size());
    for(int i = 0; i<vector.size(); ++i){
        activated_vector[i] = func_(vector[i]);
    }
    return  activated_vector;
    
    
}
Matrix ActivationFunc::GetDifferential(Vector vector){
    Vector tmp_vector(vector.size());
    
    for(int i = 0; i<vector.size(); ++i){
        tmp_vector[i] = derivative_(vector[i]);
    }
    Matrix differential = tmp_vector.asDiagonal();
    return differential;

}

}

