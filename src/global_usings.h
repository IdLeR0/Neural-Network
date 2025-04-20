#pragma once
#include <Eigen/Dense>
#include <EigenRand/EigenRand>

namespace network {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using VectorT = Eigen::RowVectorXd;
using Index = Eigen::Index;
using MatrixView = Eigen::Ref<const Matrix>;
using DataType = Eigen::MatrixXd::Scalar;

struct LayerParams {
    Matrix weights;
    Vector bias;
};

struct Data {
    Matrix input;
    Matrix output;
};

}  // namespace network