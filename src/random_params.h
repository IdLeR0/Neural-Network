#pragma once
#include "global_usings.h"

namespace network {

class RandomParams {
    using Generator = Eigen::Rand::Vmt19937_64;

public:
    RandomParams(Index seed = kDefaultSeed);
    Matrix GenerateNormalMatrix(Index rows, Index cols, DataType mean = 0, DataType stdev = 1);
    Vector GenerateNormalVector(Index rows, DataType mean = 0.0, DataType stdev = 1.0);
    Matrix GenerateUniformMatrix(Index rows, Index cols, DataType low = 0.0, DataType high = 1.0);
    Vector GenerateUniformVector(Index rows, DataType low = 0.0, DataType high = 0.0);
    Matrix GenerateConstantMatrix(Index rows, Index cols, DataType value);
    Vector GenerateConstantVector(Index rows, DataType value);

private:
    static constexpr Index kDefaultSeed = 42;
    Generator generator_{kDefaultSeed};
};
}  // namespace network