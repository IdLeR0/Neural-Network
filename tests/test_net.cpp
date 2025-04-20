#include <gtest/gtest.h>
#include <fstream>
#include <exception>
#include <random>
#include "activation_function.h"
#include "dataloader.h"
#include "file_reader_writer.h"
#include "global_usings.h"
#include "loss_function.h"
#include "mnist_utils.h"
#include "random_params.h"
using namespace network;

void ClearFile(std::filesystem::path filename) {
    std::ofstream file(filename, std::ios::binary | std::ios::trunc);
    file.close();
}

Net BuildMnistNet() {
    constexpr Index kFirstSize = 784;
    constexpr Index kSecondSize = 256;
    constexpr Index kThirdSize = 10;
    std::vector<Index> layer_sizes = {kFirstSize, kSecondSize, kThirdSize};
    std::vector<ActivationFunc::Name> activation_functions = {ActivationFunc::Name::ReLU,
                                                              ActivationFunc::Name::Softmax};
    Net nn(layer_sizes, activation_functions);
    return nn;
}

MnistUtils PrepareMnistUtils() {
    const std::filesystem::path path = "../libs/mnist";
    MnistUtils mnist_utils(path);
    return mnist_utils;
}

DataLoader PrepareDataloader(MnistUtils& mnist_utils) {
    Data train_data = mnist_utils.GetTrainData();
    DataLoader dl(std::move(train_data));
    return dl;
}

void TrainNet(Net& nn, DataLoader& dl) {
    constexpr Index kNumEpocs = 17;
    constexpr Index kBatchSize = 64;
    constexpr Info kInfo = Info::On;
    constexpr LossFunc::Name kLossFunc = LossFunc::Name::CrossEntropy;
    nn.Train(dl, kLossFunc, kBatchSize, kNumEpocs, kInfo);
}

Index GetRandomNum() {
    std::mt19937 gen(std::random_device{}());
    static constexpr Index kLow = 1;
    static constexpr Index kHigh = 1000;
    std::uniform_int_distribution<Index> dist(kLow, kHigh);
    Index random_number = dist(gen);
    return random_number;
}

ActivationFunc::Name GetRandomActivationFunc() {
    static std::vector<ActivationFunc::Name> activation_functions = {
        ActivationFunc::Name::Tanh, ActivationFunc::Name::Id, ActivationFunc::Name::ReLU,
        ActivationFunc::Name::Sigmoid, ActivationFunc::Name::Softmax};
    Index rand_num = GetRandomNum();
    return activation_functions[rand_num % activation_functions.size()];
}

Net BuildRandomNet() {
    std::vector<Index> sizes(GetRandomNum());

    for (Index i = 0; i < sizes.size(); ++i) {
        sizes[i] = GetRandomNum();
    }
    std::vector<LayerParams> layer_params;
    RandomParams rnd;
    for (Index i = 0; i < sizes.size() - 1; ++i) {
        Matrix weights = rnd.GenerateUniformMatrix(sizes[i + 1], sizes[i]);
        Vector bias = rnd.GenerateNormalVector(sizes[i + 1]);
        layer_params.emplace_back(weights, bias);
    }
    std::vector<ActivationFunc::Name> activation_functions(sizes.size() - 1);
    for (Index i = 0; i < activation_functions.size(); ++i) {
        activation_functions[i] = GetRandomActivationFunc();
    }
    Net nn(sizes, activation_functions, layer_params);
    return nn;
}

TEST(Net, ReadWrite) {
    const std::filesystem::path filename = "save_net.bin";
    constexpr Index kIter = 10;
    for (Index i = 0; i < kIter; ++i) {
        ClearFile(filename);
        FileWriter w(filename);
        Net nn = BuildRandomNet();
        w << nn;
        w.CloseFile();
        FileReader r(filename);
        Net net;
        r >> net;
        r.CloseFile();
        EXPECT_EQ(nn, net);
    }
    ClearFile(filename);
}

TEST(Net, MnistCorrection) {
    Net nn = BuildMnistNet();
    MnistUtils mnist_utils = PrepareMnistUtils();
    DataLoader dl = PrepareDataloader(mnist_utils);
    TrainNet(nn, dl);
    constexpr DataType kExpectedAccuracy = 98;
    DataType accuracy = mnist_utils.ComputeAccuracy(nn);
    EXPECT_GE(accuracy, kExpectedAccuracy) << accuracy << std::endl;
}

int main(int argc, char** argv) {
    try {
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    } catch (...) {}
}