#pragma once
#include "types.h"

class Sorter {
public:
    virtual ~Sorter() = default;
    virtual void sort(BinningOutput& binning, FrameAllocator& allocator) = 0;
};
