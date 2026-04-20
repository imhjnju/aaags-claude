// harmonyos_3dgs/src/vulkan/vk_render_main.cpp
//
// gs3d_vk_render — Vulkan forward render CLI for calibration.
// Usage: gs3d_vk_render <model.ply> <cameras.json> [cam_id]

#include "camera_utils.h"
#include "image_io.h"
#include "ply_loader.h"
#include "renderer.h"
#include "vulkan/preprocessor_vulkan.h"
#include "vulkan/rasterizer_vulkan.h"
#include "vulkan/sorter_vulkan.h"
#include "vulkan/tile_binner_vulkan.h"
#include "vulkan/vk_context.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "Usage: %s <model.ply> <cameras.json> [cam_id]\n",
                     argv[0]);
        return 1;
    }

    const char* ply_path = argv[1];
    const char* cam_path = argv[2];
    int cam_id = (argc >= 4) ? std::atoi(argv[3]) : 0;

    // 1. Vulkan init
    VulkanContext ctx;
    if (!ctx.init()) {
        std::fprintf(stderr, "[FAIL] No Vulkan device with compute queue\n");
        return 1;
    }
    std::printf("[info] Vulkan device: %s\n", ctx.deviceName().c_str());

    // 2. Load model
    std::printf("Loading model: %s\n", ply_path);
    auto model = loadPly(ply_path);
    std::printf("  %d Gaussians, SH degree %d, filter_3D: %s\n",
                model.data.count, model.data.sh_degree,
                model.data.filter_3D ? "yes" : "no");

    // 3. Load camera
    Camera cam = loadCameraJson(cam_path, cam_id);

    // 4. Render config
    RenderConfig config{};
    const char* bg_env = std::getenv("BG_WHITE");
    float bg_val = (bg_env && bg_env[0] == '1') ? 1.0f : 0.0f;
    config.bg_color[0] = config.bg_color[1] = config.bg_color[2] = bg_val;
    config.sh_degree = model.data.sh_degree;
    config.eval_3D = false;
    config.antialiasing = false;
    std::printf("Config: eval_3D=%s, bg=%.0f, sh_degree=%d\n",
                config.eval_3D ? "ON" : "OFF", bg_val, config.sh_degree);

    // 5. Create Vulkan renderer
    size_t alloc_size = 512ULL * 1024 * 1024;
    if (model.data.count > 500000)
        alloc_size = 2ULL * 1024 * 1024 * 1024;

    auto renderer = std::make_unique<Renderer>(
        std::make_unique<PreprocessorVulkan>(ctx),
        std::make_unique<TileBinnerVulkan>(ctx),
        std::make_unique<SorterVulkan>(ctx),
        std::make_unique<RasterizerVulkan>(ctx),
        alloc_size);

    // 6. Render
    int W = cam.width, H = cam.height;
    std::vector<float> image(static_cast<size_t>(W) * H * 3, 0.0f);
    std::printf("Rendering %dx%d...\n", W, H);
    auto t0 = std::chrono::high_resolution_clock::now();
    renderer->render(model.data, cam, config, image.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Render time: %.1f ms\n", ms);

    // 7. Save
    const char* out_path = "output_vk.ppm";
    writePPM(out_path, image.data(), W, H);
    std::printf("Saved → %s\n", out_path);

    model.free();
    return 0;
}
