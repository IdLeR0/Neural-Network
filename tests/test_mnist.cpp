#include <gtest/gtest.h>
#include <fstream>
#include "activation_function.h"
#include "dataloader.h"
#include "file_reader_writer.h"
#include "loss_function.h"
#include "mnist_utils.h"

void ClearFile(std::string filename) {
    std::ofstream file(filename, std::ios::binary | std::ios::trunc);
    file.close();
}

TEST(Mnist, WriteRead) {
    using namespace network;
    std::vector<Index> layer_sizes = {784, 256, 10};
    std::vector<ActivationFunc::Name> activation_functions = {ActivationFunc::Name::ReLU,
                                                              ActivationFunc::Name::Softmax};
    Net nn(layer_sizes, activation_functions);
    std::string path = "../libs/mnist";
    MnistUtils mnist_utils(path);
    Data train_data = mnist_utils.GetTrainData();
    DataLoader dl(std::move(train_data));
    constexpr Index kNumEpocs = 17;
    constexpr Index kBatchSize = 64;
    nn.Train(dl, LossFunc::Name::CrossEntropy, kBatchSize, kNumEpocs, Info::On);
    constexpr DataType kExpectedAccuracy = 0;
    DataType accuracy = mnist_utils.ComputeAccuracy(nn);
    EXPECT_GE(accuracy, kExpectedAccuracy);

    std::string filename = "save_net.bin";

    ClearFile(filename);
    FileWriter w(filename);
    w << nn;
    w.CloseFile();
    FileReader r(filename);
    Net net;
    r >> net;
    r.CloseFile();
    EXPECT_EQ(nn, net);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}