#include "activation_function.h"
#include "loss_function.h"
#include "net.h"
#include <iostream>
#include <random>

// main я пока использую, чтобы посмотреть так все работает как я ожидаю или нет. В итоговой проект
// он не войдет.
using namespace network;

void RandomMatrix(Matrix& mat) {
    for (int i = 0; i < mat.cols(); ++i) {
        for (int j = 0; j < mat.rows(); ++j) {
            std::mt19937 gen(i + j);
            std::uniform_real_distribution<double> dist(0, 100);
            mat(j, i) = dist(gen);
        }
    }
}
int main() {

    Matrix input(784, 60000);
    Matrix output(9, 60000);
    RandomMatrix(input);
    RandomMatrix(output);

    Data data{input, output};
    DataLoader dl(std::move(data));
    // std::vector<Data> dt = dl.Batches(1);
    // dl.ShuffleData();
    // dt = dl.Batches(2);

    Net nn(std::move(dl), LossFunc::Name::CrossEntropy);
    nn.AddLayer(In{784}, Out{3}, ActivationFunc::Name::ReLU);
    nn.AddLayer(In{3}, Out{7}, ActivationFunc::Name::Sigmoid);
    nn.AddLayer(In{7}, Out{9}, ActivationFunc::Name::Softmax);
    nn.Backward(60000, 1);

    // nn.Backward(2, 100);
}
