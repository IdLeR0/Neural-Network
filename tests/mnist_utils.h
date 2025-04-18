#include "net.h"
#include "mnist/mnist_reader.hpp"

namespace network {

class MnistUtils {
    using Images = std::vector<std::vector<uint8_t>>;
    using Labels = std::vector<uint8_t>;

public:
    MnistUtils(const std::string& path);
    Data GetTrainData();
    Data GetTestData();
    DataType ComputeAccuracy(const Net& net);

private:
    Matrix ConvertToMatrix(const Images& images);
    Matrix LabelToMatrix(const Labels& labels);
    static constexpr Index kOutputVectorSize = 10;
    static constexpr Index kInputVectorSize = 784;
    std::string path_;
};

}  // namespace network