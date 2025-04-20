#include "random_params.h"
#include "global_usings.h"

namespace network {
RandomParams::RandomParams(Index seed) : generator_(seed) {
}

Matrix RandomParams::GenerateNormalMatrix(Index rows, Index cols, DataType mean, DataType stdev) {
    return Eigen::Rand::normal<Matrix>(rows, cols, generator_, mean, stdev);
}

Vector RandomParams::GenerateNormalVector(Index rows, DataType mean, DataType stdev) {
    return GenerateNormalMatrix(rows, 1, mean, stdev);
}

Matrix RandomParams::GenerateUniformMatrix(Index rows, Index cols, DataType min, DataType max) {
    return Eigen::Rand::uniformReal<Matrix>(rows, cols, generator_, min, max);
}

Vector RandomParams::GenerateUniformVector(Index rows, DataType min, DataType max) {
    return GenerateUniformMatrix(rows, 1, min, max);
}

Matrix RandomParams::GenerateConstantMatrix(Index rows, Index cols, DataType value) {
    return Eigen::MatrixXd::Constant(rows, cols, value);
}

Vector RandomParams::GenerateConstantVector(Index rows, DataType value) {
    return GenerateConstantMatrix(rows, 1, value);
}
}  // namespace network