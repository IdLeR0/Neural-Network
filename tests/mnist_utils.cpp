#include "net.h"
#include <iostream>
#include "mnist_utils.h"

namespace network {
MnistUtils::MnistUtils(const std::string& path) : path_(path) {
}

Data MnistUtils::GetTrainData() {
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(path_);
    assert(!dataset.training_images.empty() && "empty dataset");
    Matrix train_images = ConvertToMatrix(dataset.training_images);
    Matrix train_labels = LabelToMatrix(dataset.training_labels);
    return {train_images, train_labels};
}

Data MnistUtils::GetTestData() {
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(path_);
    Matrix test_images = ConvertToMatrix(dataset.test_images);
    Matrix test_labels = LabelToMatrix(dataset.test_labels);
    return {test_images, test_labels};
}

Matrix MnistUtils::ConvertToMatrix(const std::vector<std::vector<uint8_t>>& images) {
    Index num_images = images.size();
    Matrix eigen_images(784, num_images);

    for (Index i = 0; i < num_images; ++i) {
        for (Index j = 0; j < kInputVectorSize; ++j) {
            eigen_images(j, i) = static_cast<DataType>(images[i][j]) / 255.0;
        }
    }

    return eigen_images;
}

Matrix MnistUtils::LabelToMatrix(const std::vector<uint8_t>& labels) {
    Index num_labels = labels.size();
    Matrix onehot(kOutputVectorSize, num_labels);
    onehot.setZero();

    for (Index i = 0; i < num_labels; ++i) {
        onehot(labels[i], i) = 1.0;
    }

    return onehot;
}

double MnistUtils::ComputeAccuracy(const Net& net) {
    Data test_data = GetTestData();
    Matrix net_ans = net.Evaluate(test_data.input);
    assert(net_ans.cols() == test_data.output.cols() && "wrong cols");
    assert(net_ans.rows() == test_data.output.rows() && "wrong rows");
    Index test_size = test_data.input.cols();
    assert(test_size != 0 && "bad test data");
    Index cnt = 0;
    Index cur_true_ans;
    Index cur_ans;
    for (Index i = 0; i < test_size; ++i) {
        net_ans.col(i).maxCoeff(&cur_ans);
        test_data.output.col(i).maxCoeff(&cur_true_ans);
        if (cur_ans == cur_true_ans) {
            ++cnt;
        }
    }
    DataType accuracy = static_cast<DataType>(cnt) / static_cast<DataType>(test_size);
    std::cout << "Accuracy:" << " " << accuracy << std::endl;
    return accuracy;
}
}  // namespace network
