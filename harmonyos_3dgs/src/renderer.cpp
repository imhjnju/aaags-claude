#include "renderer.h"
#include <chrono>
#include <cstdio>

Renderer::Renderer(std::unique_ptr<Preprocessor> pp,
                   std::unique_ptr<TileBinner> tb,
                   std::unique_ptr<Sorter> s,
                   std::unique_ptr<Rasterizer> r,
                   size_t alloc_cap)
    : preprocessor_(std::move(pp)), tile_binner_(std::move(tb)),
      sorter_(std::move(s)), rasterizer_(std::move(r)), allocator_(alloc_cap) {}

void Renderer::render(const GaussianData& g, const Camera& cam,
                      const RenderConfig& cfg, float* img, float* depth) {
    allocator_.reset();

    auto t0 = std::chrono::high_resolution_clock::now();
    auto pre = preprocessor_->process(g, cam, cfg, allocator_);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto bin = tile_binner_->bin(pre, g.count, cam, cfg, allocator_);
    auto t2 = std::chrono::high_resolution_clock::now();

    if (bin.total_pairs > 0)
        sorter_->sort(bin, allocator_);
    auto t3 = std::chrono::high_resolution_clock::now();

    rasterizer_->rasterize(pre, bin, cam, cfg, img, depth);
    auto t4 = std::chrono::high_resolution_clock::now();

    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(stderr, "  Preprocess: %.1f ms\n", ms(t0, t1));
    std::fprintf(stderr, "  TileBinner: %.1f ms\n", ms(t1, t2));
    std::fprintf(stderr, "  Sorter:     %.1f ms\n", ms(t2, t3));
    std::fprintf(stderr, "  Rasterizer: %.1f ms\n", ms(t3, t4));
    std::fprintf(stderr, "  Total:      %.1f ms\n", ms(t0, t4));
}
