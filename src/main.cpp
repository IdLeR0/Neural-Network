#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include <random>
#include <iostream>
using RandGen = Eigen::Rand::Vmt19937_64;

RandGen& GetRng() {
    static RandGen rng = 1;
    return rng;
}
int main() {
    int rows = 2;
    int cols = 2;
    Eigen::MatrixXd result = Eigen::Rand::normal<Eigen::MatrixXd>(rows, cols, GetRng());
    std::cout << result << std::endl;
    return 0;
}