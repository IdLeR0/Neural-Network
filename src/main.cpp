
#include "loss_function.h"
#include "net.h"
#include <iostream>
#include <random>
#include <utility>
#include "mnist/mnist_reader.hpp"

//за это сори, я по фасту накидывал, чтобы протестить
using namespace network;
Matrix ConvertToMatrix(const std::vector<std::vector<uint8_t>>& images) {
    size_t num_images = images.size();
    Matrix eigen_images(784, num_images);  

    for (size_t i = 0; i < num_images; ++i) {
        for (int j = 0; j < 784; ++j) {
            eigen_images(j, i) = static_cast<double>(images[i][j]) / 255.0;
        }
    }

    return eigen_images;
}
Matrix LabelToVector(const std::vector<uint8_t>& labels) {
    size_t num_labels = labels.size();
    Matrix onehot(10, num_labels);  
    onehot.setZero();

    for (size_t i = 0; i < num_labels; ++i) {
        onehot(labels[i], i) = 1.0;  
    }

    return onehot;
}

std::pair<Data, Data> GetTrainDataSet(std::string path_to_data) {
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(path_to_data);
    assert(!dataset.training_images.empty() && "empty dataset");
    Matrix train_images = ConvertToMatrix(dataset.training_images);
    Matrix train_labels = LabelToVector(dataset.training_labels);
    Data train{train_images, train_labels};
    Matrix test_images = ConvertToMatrix(dataset.test_images);
    Matrix test_labels = LabelToVector(dataset.test_labels);
    Data test_data{test_images, test_labels};
    return std::make_pair(train, test_data);
}

int main() {
    std::string path = "../libs/mnist";
    std::pair<Data, Data> ans = GetTrainDataSet(path);
    DataLoader dl(std::move(ans.first));
    Net nn(std::move(dl), LossFunc::Name::CrossEntropy);
    nn.AddLayer(In{784}, Out{256}, ActivationFunc::Name::ReLU);
    nn.AddLayer(In{256}, Out{10}, ActivationFunc::Name::Softmax);
    nn.Backward(64, 15);
    std::cout << "///////////////////////////////// TEST PATR//////////////////////////////////////"
              << std::endl;
    Matrix evalute_test = nn.Evalute(ans.second.input);
    int size_test = ans.second.input.cols();
    int cnt = 0;
    std::cout << " Size TESt       " << size_test << std::endl;
    Index max_our_ind;
    Index max_test_ind;
    for (int i = 0; i < size_test; ++i) {
        evalute_test.col(i).maxCoeff(&max_our_ind);
        ans.second.output.col(i).maxCoeff(&max_test_ind);
        if (max_test_ind == max_our_ind) {
            ++cnt;
        }
    }
    double accuracy = (static_cast<double>(cnt) / size_test) * 100;
    std::cout<< accuracy << std::endl;
}
