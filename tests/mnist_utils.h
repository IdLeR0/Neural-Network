#include "net.h"


namespace network {

class MnistUtils {
    using Images = std::vector<std::vector<uint8_t>>;
    using Labels = std::vector<uint8_t>;
    using Path = std::filesystem::path;

public:
    MnistUtils(const Path& path);
    Data GetTrainData();
    Data GetTestData();
    DataType ComputeAccuracy(const Net& net);

private:
    Matrix ConvertToMatrix(const Images& images);
    Matrix LabelToMatrix(const Labels& labels);
    static constexpr Index kOutputVectorSize = 10;
    static constexpr Index kInputVectorSize = 784;
    Path path_;
};

}  // namespace network