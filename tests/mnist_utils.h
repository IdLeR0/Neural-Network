#include "net.h"
#include "mnist/mnist_reader.hpp"

namespace network {
struct TrainTestData{
    Data train;
    Data test;
};
class MnistUtils {
    using Images = std::vector<std::vector<uint8_t>>;
    using Labels = std::vector<uint8_t>;
public:
MnistUtils(const std::string& path);
    Data GetTrainData();
    Data GetTestData();
    double ComputeAccuracy(const Net& net);

private:
    Matrix ConvertToMatrix(const Images& images);
    Matrix LabelToMatrix(const Labels& labels);
    std::string path_;
};


}  // namespace network