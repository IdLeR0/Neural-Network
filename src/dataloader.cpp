
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include "dataloader.h"

namespace network {
namespace {
using RandomGenerator = std::mt19937_64;

RandomGenerator& GetGenerator() {
    static constexpr Index kDefaultSeed = 42;
    static RandomGenerator gen(kDefaultSeed);
    return gen;
}
}  // namespace

DataLoader::DataLoader(Data&& data) {
    assert(data.input.cols() == data.output.cols() && "Data input and output columns mismatch");
    data_ = std::move(data);
}

DataLoader::DataLoader(const Data& data) {
    assert(data.input.cols() == data.output.cols() && "Data input and output columns mismatch");
    data_ = data;
}

Index DataLoader::Size() const {
    return data_.input.cols();
}

std::vector<DataView> DataLoader::Batches(Index batch_size) const {
    assert(batch_size > 0 && "Batch size must be positive");
    assert(batch_size <= Size() && "Batch size exceeds training data size");
    std::vector<DataView> batches;
    Index data_size = Size();
    batches.reserve(data_size);
    for (Index i = 0; i < data_size; i += batch_size) {
        Index cur_batch_size = std::min(batch_size, data_size - i);
        batches.push_back(std::move(GetBatch(i, cur_batch_size)));
    }

    return batches;
}

// namespace

void DataLoader::ShuffleData() {
    RandomGenerator rand_gen = GetGenerator();
    for (Index i = Size() - 1; i > 0; --i) {
        std::uniform_int_distribution<Index> uni_gen(0, i);
        Index j = uni_gen(rand_gen);
        data_.input.col(i).swap(data_.input.col(j));
        data_.output.col(i).swap(data_.output.col(j));
    }
}

Data DataLoader::GetData() const {
    return data_;
}

DataView DataLoader::GetBatch(Index begin, Index size) const {
    assert(begin >= 0 && size > 0 && "Invalid batch parameters");
    assert(begin + size <= Size() && "Batch exceeds training data");
    MatrixView inputs = data_.input.middleCols(begin, size);
    MatrixView outputs = data_.output.middleCols(begin, size);
    DataView batch{inputs, outputs};

    return batch;
}

}  // namespace network