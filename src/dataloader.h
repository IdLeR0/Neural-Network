#pragma once
#include "linalg.h"
namespace network {

struct Data {
    Matrix input;
    Matrix output;
};
namespace details {

class Shuffle {

public:
    Shuffle();
    Shuffle(int seed);
    void ShuffleData(Index begin, Index end, Data& data);

private:
    static constexpr int kDefaultSeed = 42;
    std::mt19937 gen_;
};
}  // namespace details
class DataLoader {
    using Shuffle = details::Shuffle;

public:
    DataLoader(const Data& data);
    DataLoader(Data&& data);
    int Size() const;
    std::vector<Data> Batches(int batch_size) const;
    void ShuffleData(Shuffle& rnd = GlobalShuffle());
    Data GetData() const;

private:
    static Shuffle& GlobalShuffle();
    Data GetBatch(Index begin, int size) const;
    Data data_;
};

}  // namespace network