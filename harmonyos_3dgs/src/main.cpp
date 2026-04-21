#include "ply_loader.h"
#include "renderer.h"
#include "image_io.h"
#include "cpu/preprocessor_cpu.h"
#include "cpu/tile_binner_cpu.h"
#include "cpu/sorter_cpu.h"
#include "cpu/rasterizer_cpu.h"
#ifdef ENABLE_OPENCL
#include "gpu/opencl_context.h"
#include "gpu/preprocessor_gpu.h"
#include "gpu/tile_binner_gpu.h"
#include "gpu/sorter_gpu.h"
#include "gpu/rasterizer_gpu.h"
#endif
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cfloat>
#include <fstream>
#include <string>
#include <stdexcept>
#include "camera_utils.h"

int main(int argc, char** argv) {
    // Check for --cl-info flag (can appear anywhere)
#ifdef ENABLE_OPENCL
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--cl-info") == 0) {
            OpenCLContext cl_ctx;
            if (!cl_ctx.init()) {
                printf("OpenCL initialization failed.\n");
                return 1;
            }
            printf("OpenCL device info:\n");
            printf("  Max work-group size: %zu\n", cl_ctx.maxWorkGroupSize());
            printf("  Max local memory:    %zu bytes\n", cl_ctx.maxLocalMemSize());
            printf("  Max compute units:   %u\n", cl_ctx.maxComputeUnits());
            printf("  Max alloc size:      %zu MB\n", cl_ctx.maxAllocSize() / (1024 * 1024));
            cl_ctx.release();
            return 0;
        }
    }
#endif

    if (argc < 2) {
        printf("Usage: %s <model.ply> [cameras.json] [cam_id] [--gpu] [--cl-info]\n", argv[0]);
        return 1;
    }

    // Detect --gpu flag (can appear anywhere after argv[0])
    bool use_gpu = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--gpu") == 0) {
            use_gpu = true;
        }
    }

#ifndef ENABLE_OPENCL
    if (use_gpu) {
        printf("Warning: --gpu requested but ENABLE_OPENCL not compiled in. Falling back to CPU.\n");
        use_gpu = false;
    }
#endif

#ifdef ENABLE_OPENCL
    OpenCLContext cl_ctx;
    if (use_gpu) {
        if (!cl_ctx.init()) {
            printf("OpenCL init failed, falling back to CPU\n");
            use_gpu = false;
        } else {
            printf("OpenCL GPU backend initialized (compute units: %u, max WG: %zu)\n",
                   cl_ctx.maxComputeUnits(), cl_ctx.maxWorkGroupSize());
        }
    }
#endif

    printf("Loading model: %s\n", argv[1]);
    auto model = loadPly(argv[1]);
    printf("Loaded %d Gaussians, SH degree %d, filter_3D: %s\n",
           model.data.count, model.data.sh_degree,
           model.data.filter_3D ? "yes" : "no");

    Camera cam;
    int width, height;
    // Check if second arg is a .json file or a number (width)
    // Skip --gpu and --cl-info flags when checking positional args
    bool use_json = false;
    if (argc >= 3) {
        std::string arg2(argv[2]);
        if (arg2 != "--gpu" && arg2 != "--cl-info")
            use_json = (arg2.size() > 5 && arg2.substr(arg2.size()-5) == ".json");
    }

    if (use_json) {
        // Find cam_id: first numeric arg after argv[2], skipping flags
        int cam_id = 0;
        for (int i = 3; i < argc; i++) {
            if (std::strcmp(argv[i], "--gpu") != 0 && std::strcmp(argv[i], "--cl-info") != 0) {
                cam_id = atoi(argv[i]);
                break;
            }
        }
        cam = loadCameraJson(argv[2], cam_id);
        width = cam.width;
        height = cam.height;
    } else {
        // Find width/height: first two numeric args after argv[1], skipping flags
        int pos_args[2] = {800, 600};
        int pos_idx = 0;
        for (int i = 2; i < argc && pos_idx < 2; i++) {
            if (std::strcmp(argv[i], "--gpu") != 0 && std::strcmp(argv[i], "--cl-info") != 0) {
                pos_args[pos_idx++] = atoi(argv[i]);
            }
        }
        width = pos_args[0];
        height = pos_args[1];
        cam = autoCamera(model.data, width, height);
    }

    RenderConfig config{};
    // Default black background; set BG_WHITE=1 for white
    const char* bg_env = getenv("BG_WHITE");
    float bg_val = (bg_env && bg_env[0] == '1') ? 1.0f : 0.0f;
    config.bg_color[0] = config.bg_color[1] = config.bg_color[2] = bg_val;
    const char* sh_env = getenv("SH_DEGREE");
    config.sh_degree = sh_env ? atoi(sh_env) : model.data.sh_degree;
    const char* aa_env = getenv("AA");
    config.antialiasing = (aa_env && aa_env[0] == '1');  // AA=1 to enable, default OFF
    // Auto-enable eval_3D if model has filter_3D (AAA-Gaussians model)
    // Can be overridden with EVAL_3D=0 or EVAL_3D=1
    const char* eval3d_env = getenv("EVAL_3D");
    if (eval3d_env)
        config.eval_3D = (eval3d_env[0] == '1');
    else
        config.eval_3D = (model.data.filter_3D != nullptr);
    // TRAINING=1 disables upper SH color clamp (matches Python training mode).
    const char* train_env = getenv("TRAINING");
    if (train_env && train_env[0] == '1')
        config.training = true;
    printf("Antialiasing: %s, eval_3D: %s, training: %s\n",
           config.antialiasing ? "ON" : "OFF",
           config.eval_3D ? "ON" : "OFF",
           config.training ? "ON" : "OFF");

    // Diagnostic: dump first few Gaussians' raw data for cross-platform comparison
    if (getenv("DUMP_DIAG")) {
        printf("=== DIAGNOSTIC: Model data (first 3 Gaussians) ===\n");
        for (int i = 0; i < 3 && i < model.data.count; i++) {
            printf("G[%d] pos=(%.8f,%.8f,%.8f)\n", i,
                model.data.positions[i*3], model.data.positions[i*3+1], model.data.positions[i*3+2]);
            printf("  scale=(%.8f,%.8f,%.8f) rot=(%.8f,%.8f,%.8f,%.8f)\n",
                model.data.scales[i*3], model.data.scales[i*3+1], model.data.scales[i*3+2],
                model.data.rotations[i*4], model.data.rotations[i*4+1],
                model.data.rotations[i*4+2], model.data.rotations[i*4+3]);
            printf("  opacity=%.8f sh_dc=(%.8f,%.8f,%.8f)\n",
                model.data.opacities[i], model.data.sh_coeffs[i*model.data.max_coeffs*3],
                model.data.sh_coeffs[i*model.data.max_coeffs*3+1],
                model.data.sh_coeffs[i*model.data.max_coeffs*3+2]);
        }
        printf("=== Camera ===\n");
        printf("  view[0..3]=%.8f,%.8f,%.8f,%.8f\n", cam.view_matrix[0], cam.view_matrix[1], cam.view_matrix[2], cam.view_matrix[3]);
        printf("  view[4..7]=%.8f,%.8f,%.8f,%.8f\n", cam.view_matrix[4], cam.view_matrix[5], cam.view_matrix[6], cam.view_matrix[7]);
        printf("  viewproj[0..3]=%.8f,%.8f,%.8f,%.8f\n", cam.viewproj_matrix[0], cam.viewproj_matrix[1], cam.viewproj_matrix[2], cam.viewproj_matrix[3]);
        printf("  cam_pos=(%.8f,%.8f,%.8f) tan_fov=(%.8f,%.8f)\n",
            cam.cam_pos[0], cam.cam_pos[1], cam.cam_pos[2], cam.tan_fovx, cam.tan_fovy);
    }

    size_t alloc_size = 512ULL * 1024 * 1024;
    if (model.data.count > 500000)
        alloc_size = 2ULL * 1024 * 1024 * 1024;

    printf("Creating renderer with %zu MB allocator (%s backend)\n",
           alloc_size / (1024 * 1024), use_gpu ? "GPU" : "CPU");

    // Construct renderer with appropriate backend
    std::unique_ptr<Renderer> renderer;
#ifdef ENABLE_OPENCL
    if (use_gpu) {
        renderer = std::make_unique<Renderer>(
            std::make_unique<PreprocessorGPU>(cl_ctx),
            std::make_unique<TileBinnerGPU>(cl_ctx),
            std::make_unique<SorterGPU>(cl_ctx),
            std::make_unique<RasterizerGPU>(cl_ctx),
            alloc_size);
    } else
#endif
    {
        renderer = std::make_unique<Renderer>(
            std::make_unique<PreprocessorCPU>(),
            std::make_unique<TileBinnerCPU>(),
            std::make_unique<SorterCPU>(),
            std::make_unique<RasterizerCPU>(),
            alloc_size);
    }

    std::vector<float> image(width * height * 3, 0.0f);

    printf("Rendering %dx%d...\n", width, height);
    auto start = std::chrono::high_resolution_clock::now();
    renderer->render(model.data, cam, config, image.data());
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Render time: %.1f ms\n", ms);

    // Check for multi-angle mode: if env var MULTI_ANGLE is set, render from multiple views
    const char* multi = getenv("MULTI_ANGLE");
    if (multi && !use_json) {
        // Render from 6 angles around the scene
        float cx = cam.cam_pos[0], cy = cam.cam_pos[1], cz = cam.cam_pos[2];
        // Recompute scene center from auto camera logic
        std::vector<float> xs2(model.data.count), ys2(model.data.count), zs2(model.data.count);
        for (int i = 0; i < model.data.count; i++) {
            xs2[i] = model.data.positions[i*3];
            ys2[i] = model.data.positions[i*3+1];
            zs2[i] = model.data.positions[i*3+2];
        }
        std::sort(xs2.begin(), xs2.end());
        std::sort(ys2.begin(), ys2.end());
        std::sort(zs2.begin(), zs2.end());
        int p5 = model.data.count * 5 / 100, p95 = model.data.count * 95 / 100;
        float scx = (xs2[p5]+xs2[p95])*0.5f, scy = (ys2[p5]+ys2[p95])*0.5f, scz = (zs2[p5]+zs2[p95])*0.5f;
        float span_x = xs2[p95]-xs2[p5], span_y = ys2[p95]-ys2[p5], span_z = zs2[p95]-zs2[p5];
        float srad = sqrtf(span_x*span_x+span_y*span_y+span_z*span_z)*0.5f;
        float dist2 = srad / cam.tan_fovx * 0.8f;

        // 6 viewpoints: front, back, left, right, top-front, bottom-front
        struct ViewDef { const char* name; float dx, dy, dz; };
        ViewDef views[] = {
            {"front",        0.0f,  0.0f, -1.0f},
            {"back",         0.0f,  0.0f,  1.0f},
            {"left",        -1.0f,  0.0f,  0.0f},
            {"right",        1.0f,  0.0f,  0.0f},
            {"top_front",    0.0f,  0.8f, -0.6f},
            {"diagonal",     0.6f,  0.3f, -0.7f},
        };

        for (auto& v : views) {
            float len = sqrtf(v.dx*v.dx + v.dy*v.dy + v.dz*v.dz);
            float eye[3] = {scx + dist2*v.dx/len, scy + dist2*v.dy/len, scz + dist2*v.dz/len};
            float center[3] = {scx, scy, scz};
            float up[3] = {0, 1, 0};
            // For top view, use different up
            if (fabsf(v.dy) > 0.9f) { up[1] = 0; up[2] = -1; }

            cam.cam_pos[0] = eye[0]; cam.cam_pos[1] = eye[1]; cam.cam_pos[2] = eye[2];
            buildLookAt(eye, center, up, cam.view_matrix);
            float proj[16];
            buildPerspective(cam.tan_fovx, cam.tan_fovy, 0.01f, 1000.0f, proj);
            mat4Mul(proj, cam.view_matrix, cam.viewproj_matrix);

            std::fill(image.begin(), image.end(), 0.0f);
            printf("Rendering %s view...\n", v.name);
            auto s = std::chrono::high_resolution_clock::now();
            renderer->render(model.data, cam, config, image.data());
            auto e = std::chrono::high_resolution_clock::now();
            printf("  %.1f ms\n", std::chrono::duration<double, std::milli>(e-s).count());

            char fname[256];
            snprintf(fname, sizeof(fname), "output_%s.ppm", v.name);
            writePPM(fname, image.data(), width, height);
            printf("  Saved %s\n", fname);
        }
    } else {
        const char* outpath = "output.ppm";
        writePPM(outpath, image.data(), width, height);
        printf("Saved to %s\n", outpath);
    }

    model.free();
    return 0;
}
