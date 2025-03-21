#include "dataloader.h"
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>

namespace network {
namespace details {
Shuffle::Shuffle() : gen_(kDefaultSeed) {
}
Shuffle::Shuffle(int seed) : gen_(kDefaultSeed) {
}
void Shuffle::ShuffleData(Index begin, Index end, Data& data) {
    assert(begin >= 0 && "Negative start index");
    assert(end > begin && "Invalid range");
    assert(end <= data.input.cols() && "Range exceeds matrix columns");
    assert(data.input.cols() == data.output.cols() && "Matrix size mismatch");

    for (Index i = end - 1; i > begin; --i) {
        std::uniform_int_distribution<Index> uni(begin, i);
        Index j = uni(gen_);
        data.input.col(i).swap(data.input.col(j));
        data.output.col(i).swap(data.output.col(j));
    }
}

}  // namespace details

DataLoader::DataLoader(Data&& data) {
    assert(data.input.cols() == data.output.cols() && "Data input and output columns mismatch");
    data_ = std::move(data);
    ShuffleData();
}

DataLoader::DataLoader(const Data& data) {
    assert(data.input.cols() == data.output.cols() && "Data input and output columns mismatch");
    data_ = data;
    ShuffleData();
}
int DataLoader::Size() const {
    return data_.input.cols();
}

std::vector<Data> DataLoader::Batches(int batch_size) const {
    assert(batch_size > 0 && "Batch size must be positive");
    assert(batch_size <= Size() && "Batch size exceeds training data size");
    std::vector<Data> batches;
    int data_size = Size();
    for (int i = 0; i < data_size; i += batch_size) {
        int cur_batch_size = std::min(batch_size, data_size - i);
        batches.push_back(GetBatch(i, cur_batch_size));
    }
    return batches;
}

// namespace

void DataLoader::ShuffleData(Shuffle& rnd) {
    rnd.ShuffleData(0, Size(), data_);
}
DataLoader::Shuffle& DataLoader::GlobalShuffle() {
    static Shuffle rnd;
    return rnd;
}
Data DataLoader::GetData() const {
    return data_;
}

Data DataLoader::GetBatch(Index begin, int size) const {
    assert(begin >= 0 && size > 0 && "Invalid batch parameters");
    assert(begin + size <= Size() && "Batch exceeds training data");
    Data batch;
    batch.input = data_.input.middleCols(begin, size);
    batch.output = data_.output.middleCols(begin, size);
    return batch;
}

}  // namespace network