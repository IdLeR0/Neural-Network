#pragma once
#include "global_usings.h"

namespace network {

struct DataView {
    MatrixView input;
    MatrixView output;
};

class DataLoader {
public:
    DataLoader(const Data& data);
    DataLoader(Data&& data);
    Index Size() const;
    std::vector<DataView> Batches(Index batch_size) const;
    void ShuffleData();
    Data GetData() const;

private:
    DataView GetBatch(Index begin, Index size) const;
    Data data_;
};

}  // namespace network