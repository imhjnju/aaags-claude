#include "types.h"
#include <cstdlib>
#include <new>

FrameAllocator::FrameAllocator(size_t capacity)
    : capacity_((capacity + 63) & ~size_t(63)), offset_(0) {
    buffer_ = static_cast<char*>(std::aligned_alloc(64, capacity_));
    if (!buffer_) throw std::bad_alloc();
}

FrameAllocator::~FrameAllocator() { std::free(buffer_); }

void* FrameAllocator::allocate(size_t bytes, size_t alignment) {
    size_t aligned = (offset_ + alignment - 1) & ~(alignment - 1);
    if (aligned + bytes > capacity_)
        throw std::runtime_error("FrameAllocator out of memory");
    void* ptr = buffer_ + aligned;
    offset_ = aligned + bytes;
    return ptr;
}

void FrameAllocator::reset() { offset_ = 0; }

void FrameAllocator::grow(size_t new_capacity) {
    new_capacity = (new_capacity + 63) & ~size_t(63);
    if (new_capacity <= capacity_) return;
    std::free(buffer_);
    buffer_ = static_cast<char*>(std::aligned_alloc(64, new_capacity));
    if (!buffer_) throw std::bad_alloc();
    capacity_ = new_capacity;
    offset_ = 0;
}
size_t FrameAllocator::used() const { return offset_; }
size_t FrameAllocator::capacity() const { return capacity_; }
