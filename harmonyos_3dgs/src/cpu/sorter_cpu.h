#pragma once
#include "sorter.h"

class SorterCPU : public Sorter {
public:
    void sort(BinningOutput& binning, FrameAllocator& allocator) override;
};
