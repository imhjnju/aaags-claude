#pragma once
#include "types.h"
#include "preprocessor.h"
#include "tile_binner.h"
#include "sorter.h"
#include "rasterizer.h"
#include <memory>

class Renderer {
public:
    Renderer(std::unique_ptr<Preprocessor> pp,
             std::unique_ptr<TileBinner> tb,
             std::unique_ptr<Sorter> s,
             std::unique_ptr<Rasterizer> r,
             size_t allocator_capacity = 1024ULL * 1024 * 1024);

    void render(const GaussianData& gaussians,
                const Camera& camera,
                const RenderConfig& config,
                float* output_image,
                float* output_depth = nullptr);

private:
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<TileBinner> tile_binner_;
    std::unique_ptr<Sorter> sorter_;
    std::unique_ptr<Rasterizer> rasterizer_;
    FrameAllocator allocator_;
};
