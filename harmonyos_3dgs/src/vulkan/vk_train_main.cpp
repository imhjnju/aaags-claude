#include "ply_loader.h"
#include "image_io.h"
#include "vulkan/vk_context.h"
#include "vulkan_trainer.h"
#include "vulkan/vk_train_args.h"
#include "train_types.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <climits>
#include <cmath>
#include <vector>
#include <chrono>
#include <string>
#include <random>
#include <fstream>
#include <memory>
#include <algorithm>
#include <stdexcept>

// ---------------------------------------------------------------------------
// PLY-based initialization: convert activated GaussianData -> RawGaussianParams
// ---------------------------------------------------------------------------
static float inverse_sigmoid(float y) {
    y = std::max(1e-6f, std::min(1.0f - 1e-6f, y));
    return std::log(y / (1.0f - y));
}

static RawGaussianParams initRawFromGaussianData(const GaussianData& g) {
    RawGaussianParams raw;
    raw.count = g.count;
    raw.sh_degree = g.sh_degree;
    raw.max_coeffs = g.max_coeffs;
    int N = g.count;
    int mc3 = g.max_coeffs * 3;

    raw.raw_positions  = new float[N * 3];
    raw.raw_scales     = new float[N * 3];
    raw.raw_rotations  = new float[N * 4];
    raw.raw_sh_coeffs  = new float[N * mc3];
    raw.raw_opacities  = new float[N];

    std::memcpy(raw.raw_positions, g.positions, N * 3 * sizeof(float));

    for (int i = 0; i < N * 3; i++)
        raw.raw_scales[i] = std::log(std::max(1e-12f, g.scales[i]));

    std::memcpy(raw.raw_rotations, g.rotations, N * 4 * sizeof(float));
    std::memcpy(raw.raw_sh_coeffs, g.sh_coeffs, N * mc3 * sizeof(float));

    for (int i = 0; i < N; i++)
        raw.raw_opacities[i] = inverse_sigmoid(g.opacities[i]);

    return raw;
}

static void freeRawParams(RawGaussianParams& raw) {
    delete[] raw.raw_positions;
    delete[] raw.raw_scales;
    delete[] raw.raw_rotations;
    delete[] raw.raw_sh_coeffs;
    delete[] raw.raw_opacities;
    raw.raw_positions = raw.raw_scales = nullptr;
    raw.raw_rotations = nullptr;
    raw.raw_sh_coeffs = nullptr;
    raw.raw_opacities = nullptr;
}

// ---------------------------------------------------------------------------
// PLY saver: write RawGaussianParams to binary PLY
// ---------------------------------------------------------------------------
static bool savePly(const char* path, const RawGaussianParams& raw, const float* filter_3D = nullptr) {
    int N = raw.count;
    int mc3 = raw.max_coeffs * 3;
    int sh_rest_per_channel = raw.max_coeffs - 1;
    int sh_rest_total = sh_rest_per_channel * 3;
    int props_per_vertex = 3 + 3 + sh_rest_total + 1 + 3 + 4 + (filter_3D ? 1 : 0);

    FILE* f = fopen(path, "wb");
    if (!f) return false;

    fprintf(f, "ply\n");
    fprintf(f, "format binary_little_endian 1.0\n");
    fprintf(f, "element vertex %d\n", N);
    fprintf(f, "property float x\n");
    fprintf(f, "property float y\n");
    fprintf(f, "property float z\n");
    fprintf(f, "property float f_dc_0\n");
    fprintf(f, "property float f_dc_1\n");
    fprintf(f, "property float f_dc_2\n");
    for (int i = 0; i < sh_rest_total; i++)
        fprintf(f, "property float f_rest_%d\n", i);
    fprintf(f, "property float opacity\n");
    fprintf(f, "property float scale_0\n");
    fprintf(f, "property float scale_1\n");
    fprintf(f, "property float scale_2\n");
    fprintf(f, "property float rot_0\n");
    fprintf(f, "property float rot_1\n");
    fprintf(f, "property float rot_2\n");
    fprintf(f, "property float rot_3\n");
    if (filter_3D)
        fprintf(f, "property float filter_3D\n");
    fprintf(f, "end_header\n");

    std::vector<float> vertex(props_per_vertex);
    for (int i = 0; i < N; i++) {
        int vi = 0;
        vertex[vi++] = raw.raw_positions[i*3+0];
        vertex[vi++] = raw.raw_positions[i*3+1];
        vertex[vi++] = raw.raw_positions[i*3+2];

        const float* sh_i = raw.raw_sh_coeffs + (size_t)i * mc3;
        vertex[vi++] = sh_i[0];
        vertex[vi++] = sh_i[1];
        vertex[vi++] = sh_i[2];

        for (int c = 0; c < 3; c++)
            for (int k = 0; k < sh_rest_per_channel; k++)
                vertex[vi++] = sh_i[3 + k*3 + c];

        vertex[vi++] = raw.raw_opacities[i];
        vertex[vi++] = raw.raw_scales[i*3+0];
        vertex[vi++] = raw.raw_scales[i*3+1];
        vertex[vi++] = raw.raw_scales[i*3+2];
        vertex[vi++] = raw.raw_rotations[i*4+0];
        vertex[vi++] = raw.raw_rotations[i*4+1];
        vertex[vi++] = raw.raw_rotations[i*4+2];
        vertex[vi++] = raw.raw_rotations[i*4+3];
        if (filter_3D)
            vertex[vi++] = filter_3D[i];

        fwrite(vertex.data(), sizeof(float), props_per_vertex, f);
    }

    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// Camera matrix utilities (local — not from camera_utils.h)
// ---------------------------------------------------------------------------
static void buildLookAt(const float eye[3], const float center[3], const float up[3],
                        float view_matrix[16]) {
    float f[3] = {center[0]-eye[0], center[1]-eye[1], center[2]-eye[2]};
    float flen = sqrtf(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
    f[0]/=flen; f[1]/=flen; f[2]/=flen;

    float r[3] = {
        f[1]*up[2] - f[2]*up[1],
        f[2]*up[0] - f[0]*up[2],
        f[0]*up[1] - f[1]*up[0]
    };
    float rlen = sqrtf(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    r[0]/=rlen; r[1]/=rlen; r[2]/=rlen;

    float u[3] = {
        r[1]*f[2] - r[2]*f[1],
        r[2]*f[0] - r[0]*f[2],
        r[0]*f[1] - r[1]*f[0]
    };

    view_matrix[0]  = r[0];  view_matrix[1]  = u[0];  view_matrix[2]  = f[0]; view_matrix[3]  = 0;
    view_matrix[4]  = r[1];  view_matrix[5]  = u[1];  view_matrix[6]  = f[1]; view_matrix[7]  = 0;
    view_matrix[8]  = r[2];  view_matrix[9]  = u[2];  view_matrix[10] = f[2]; view_matrix[11] = 0;
    view_matrix[12] = -(r[0]*eye[0] + r[1]*eye[1] + r[2]*eye[2]);
    view_matrix[13] = -(u[0]*eye[0] + u[1]*eye[1] + u[2]*eye[2]);
    view_matrix[14] = -(f[0]*eye[0] + f[1]*eye[1] + f[2]*eye[2]);
    view_matrix[15] = 1;
}

static void buildPerspective(float tan_fovx, float tan_fovy, float znear, float zfar,
                              float proj[16]) {
    // Matches camera_utils.cpp and AAA-Gaussians getProjectionMatrix().
    memset(proj, 0, 16 * sizeof(float));
    proj[0]  = 1.0f / tan_fovx;
    proj[5]  = 1.0f / tan_fovy;
    proj[10] = zfar / (zfar - znear);
    proj[11] = 1.0f;
    proj[14] = -(zfar * znear) / (zfar - znear);
}

static void mat4Mul(const float A[16], const float B[16], float out[16]) {
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++) {
            float sum = 0;
            for (int k = 0; k < 4; k++)
                sum += A[k*4 + row] * B[col*4 + k];
            out[col*4 + row] = sum;
        }
}

// Compute out = A * B^T (column-major).  CUDA's effective projmatrix is
// P * w2v^T because Python transposes w2v before passing to CUDA, and
// loadMatrix4x4 un-transposes.  The viewmatrix (for p_view / W) uses
// w2v directly, but viewproj must use w2v^T to match CUDA's convention.
static void mat4MulTransposed(const float A[16], const float B[16], float out[16]) {
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++) {
            float sum = 0;
            for (int k = 0; k < 4; k++)
                sum += A[k*4 + row] * B[k*4 + col];  // B^T[col][k] = B[k][col]
            out[col*4 + row] = sum;
        }
}

static Camera buildCamera(float cam_x, float cam_y, float cam_z,
                           float fov_deg, int width, int height,
                           float look_x, float look_y, float look_z) {
    Camera cam{};
    cam.width = width;
    cam.height = height;
    cam.tan_fovx = tanf(fov_deg * 0.5f * 3.14159265f / 180.0f);
    cam.tan_fovy = cam.tan_fovx * (float)height / (float)width;
    cam.cam_pos[0] = cam_x;
    cam.cam_pos[1] = cam_y;
    cam.cam_pos[2] = cam_z;

    float eye[3] = {cam_x, cam_y, cam_z};
    float center[3] = {look_x, look_y, look_z};
    float up[3] = {0, 1, 0};

    buildLookAt(eye, center, up, cam.view_matrix);

    float proj[16];
    buildPerspective(cam.tan_fovx, cam.tan_fovy, 0.01f, 1000.0f, proj);
    // viewproj = P * w2v (matches CUDA's effective projmatrix = P @ w2v).
    mat4Mul(proj, cam.view_matrix, cam.viewproj_matrix);

    return cam;
}

static Camera autoCamera(const GaussianData& g, int width, int height) {
    std::vector<float> xs(g.count), ys(g.count), zs(g.count);
    for (int i = 0; i < g.count; i++) {
        xs[i] = g.positions[i*3];
        ys[i] = g.positions[i*3+1];
        zs[i] = g.positions[i*3+2];
    }
    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());
    std::sort(zs.begin(), zs.end());

    int p5 = g.count * 5 / 100;
    int p95 = g.count * 95 / 100;
    if (p5 >= g.count) p5 = 0;
    if (p95 >= g.count) p95 = g.count - 1;
    float cx = (xs[p5]+xs[p95])*0.5f;
    float cy = (ys[p5]+ys[p95])*0.5f;
    float cz = (zs[p5]+zs[p95])*0.5f;
    float span_x = xs[p95]-xs[p5], span_y = ys[p95]-ys[p5], span_z = zs[p95]-zs[p5];
    float srad = sqrtf(span_x*span_x + span_y*span_y + span_z*span_z) * 0.5f;

    float fov_deg = 50.0f;
    float tan_fov = tanf(fov_deg * 0.5f * 3.14159265f / 180.0f);
    float dist = srad / tan_fov * 0.8f;
    float ex = cx + dist*0.6f, ey = cy + dist*0.3f, ez = cz - dist*0.7f;
    return buildCamera(ex, ey, ez, fov_deg, width, height, cx, cy, cz);
}

// ---------------------------------------------------------------------------
// Multi-camera JSON loading
// ---------------------------------------------------------------------------
struct CameraWithImage {
    Camera cam;
    std::string img_name;
};

static std::vector<int> loadViewSchedule(const char* path) {
    std::ifstream sf(path);
    if (!sf.is_open())
        throw std::runtime_error(std::string("Cannot open view schedule ") + path);

    std::vector<int> schedule;
    int idx = 0;
    while (sf >> idx)
        schedule.push_back(idx);
    if (!sf.eof())
        throw std::runtime_error(std::string("Invalid integer in view schedule ") + path);
    return schedule;
}

static std::vector<CameraWithImage> loadCamerasJson(const char* json_path) {
    std::ifstream jf(json_path);
    if (!jf.is_open())
        throw std::runtime_error(std::string("Cannot open ") + json_path);

    std::string content((std::istreambuf_iterator<char>(jf)),
                         std::istreambuf_iterator<char>());

    auto findNum = [&](const std::string& s, size_t pos) -> std::pair<double, size_t> {
        while (pos < s.size() && (s[pos]==' '||s[pos]==','||s[pos]=='\n'||s[pos]=='\r'))
            pos++;
        size_t end = pos;
        while (end < s.size() && (s[end]=='-'||s[end]=='.'||s[end]=='e'||s[end]=='E'
               ||s[end]=='+'||(s[end]>='0'&&s[end]<='9')))
            end++;
        return {std::stod(s.substr(pos, end - pos)), end};
    };

    std::vector<CameraWithImage> cameras;
    size_t search_pos = 0;

    while (true) {
        size_t id_pos = content.find("\"id\"", search_pos);
        if (id_pos == std::string::npos) break;

        size_t obj_start = content.rfind('{', id_pos);
        int depth = 1;
        size_t p = obj_start + 1;
        while (p < content.size() && depth > 0) {
            if (content[p] == '{') depth++;
            else if (content[p] == '}') depth--;
            p++;
        }
        std::string obj = content.substr(obj_start, p - obj_start);
        search_pos = p;

        auto parseIntField = [&](const std::string& field) -> int {
            size_t fp = obj.find("\"" + field + "\"");
            if (fp == std::string::npos) return 0;
            fp = obj.find(':', fp) + 1;
            return (int)findNum(obj, fp).first;
        };
        auto parseFloatField = [&](const std::string& field) -> float {
            size_t fp = obj.find("\"" + field + "\"");
            if (fp == std::string::npos) return 0.0f;
            fp = obj.find(':', fp) + 1;
            return (float)findNum(obj, fp).first;
        };
        auto parseStringField = [&](const std::string& field) -> std::string {
            size_t fp = obj.find("\"" + field + "\"");
            if (fp == std::string::npos) return "";
            fp = obj.find(':', fp) + 1;
            size_t q1 = obj.find('"', fp);
            if (q1 == std::string::npos) return "";
            size_t q2 = obj.find('"', q1 + 1);
            if (q2 == std::string::npos) return "";
            return obj.substr(q1 + 1, q2 - q1 - 1);
        };

        CameraWithImage entry;
        int cam_id = parseIntField("id");
        entry.cam.width = parseIntField("width");
        entry.cam.height = parseIntField("height");
        float fx = parseFloatField("fx");
        float fy = parseFloatField("fy");
        if (fx <= 0 || fy <= 0) continue;
        entry.cam.tan_fovx = (float)entry.cam.width / (2.0f * fx);
        entry.cam.tan_fovy = (float)entry.cam.height / (2.0f * fy);

        size_t pos_start = obj.find("\"position\"");
        if (pos_start == std::string::npos) continue;
        pos_start = obj.find('[', pos_start) + 1;
        auto [px, p1] = findNum(obj, pos_start);
        auto [py, p2] = findNum(obj, p1);
        auto [pz, p3] = findNum(obj, p2);
        entry.cam.cam_pos[0] = (float)px;
        entry.cam.cam_pos[1] = (float)py;
        entry.cam.cam_pos[2] = (float)pz;

        float R[3][3];
        size_t rot_start = obj.find("\"rotation\"");
        if (rot_start == std::string::npos) continue;
        rot_start = obj.find('[', rot_start) + 1;
        for (int row = 0; row < 3; row++) {
            rot_start = obj.find('[', rot_start) + 1;
            for (int col = 0; col < 3; col++) {
                auto [val, next] = findNum(obj, rot_start);
                R[row][col] = (float)val;
                rot_start = next;
            }
            rot_start = obj.find(']', rot_start) + 1;
        }

        float t[3] = {
            -(R[0][0]*entry.cam.cam_pos[0] + R[1][0]*entry.cam.cam_pos[1] + R[2][0]*entry.cam.cam_pos[2]),
            -(R[0][1]*entry.cam.cam_pos[0] + R[1][1]*entry.cam.cam_pos[1] + R[2][1]*entry.cam.cam_pos[2]),
            -(R[0][2]*entry.cam.cam_pos[0] + R[1][2]*entry.cam.cam_pos[1] + R[2][2]*entry.cam.cam_pos[2])
        };

        entry.cam.view_matrix[0]=R[0][0]; entry.cam.view_matrix[1]=R[0][1]; entry.cam.view_matrix[2]=R[0][2]; entry.cam.view_matrix[3]=0;
        entry.cam.view_matrix[4]=R[1][0]; entry.cam.view_matrix[5]=R[1][1]; entry.cam.view_matrix[6]=R[1][2]; entry.cam.view_matrix[7]=0;
        entry.cam.view_matrix[8]=R[2][0]; entry.cam.view_matrix[9]=R[2][1]; entry.cam.view_matrix[10]=R[2][2]; entry.cam.view_matrix[11]=0;
        entry.cam.view_matrix[12]=t[0]; entry.cam.view_matrix[13]=t[1]; entry.cam.view_matrix[14]=t[2]; entry.cam.view_matrix[15]=1;

        float proj[16];
        buildPerspective(entry.cam.tan_fovx, entry.cam.tan_fovy, 0.01f, 100.0f, proj);
        // viewproj = P * w2v (matches CUDA's effective projmatrix = P @ w2v).
        mat4Mul(proj, entry.cam.view_matrix, entry.cam.viewproj_matrix);

        entry.img_name = parseStringField("img_name");

        printf("  Camera %d: %dx%d, pos=(%.3f,%.3f,%.3f)%s\n",
               cam_id, entry.cam.width, entry.cam.height,
               entry.cam.cam_pos[0], entry.cam.cam_pos[1], entry.cam.cam_pos[2],
               entry.img_name.empty() ? "" : (", img=" + entry.img_name).c_str());

        cameras.push_back(std::move(entry));
    }

    return cameras;
}

static void printUsage(const char* prog) {
    printf("Usage: %s --ply <input.ply> [options]\n\n", prog);
    printf("Single-camera training:\n");
    printf("  %s --ply model.ply --gt target.ppm --output trained.ply\n\n", prog);
    printf("Multi-view training:\n");
    printf("  %s --ply model.ply --cameras cameras.json --gt_dir images/ --output trained.ply\n\n", prog);
    printf("Options:\n");
    printf("  --ply <path>          Input PLY file (required)\n");
    printf("  --gt <path>           Ground truth image (PPM, single-camera mode)\n");
    printf("  --cameras <path>      Cameras JSON file (multi-view mode)\n");
    printf("  --gt_dir <path>       Directory of GT images for multi-view\n");
    printf("  --view_schedule <path>  Whitespace-separated 0-based view index per iteration\n");
    printf("  --require_all_gt <0|1>  Fail if any camera has no GT image (default: 0)\n");
    printf("  --output <path>       Output PLY file (default: trained.ply)\n");
    printf("  --iterations <N>      Training iterations (default: 30000)\n");
    printf("  --width <W>           Image width (default: 512)\n");
    printf("  --height <H>          Image height (default: 512)\n");
    printf("  --cam_x/y/z <val>     Camera position (single-camera mode)\n");
    printf("  --look_x/y/z <val>    Look-at target (default: 0,0,5)\n");
    printf("  --fov <deg>           Field of view in degrees (default: 50)\n");
    printf("  --sh_degree <0-3>     SH degree override (default: from PLY)\n");
    printf("  --save_every <N>      Checkpoint interval (default: end only)\n");
    printf("  --log_every <N>       Print loss interval (default: 100)\n");
    printf("  --densify <0|1>       Enable densification schedule; defaults to MCMC growth (default: 0)\n");
    printf("  --cap_max <N>         MCMC Gaussian cap; >0 also enables densification, 0 with --densify 1 selects legacy\n");
    printf("  --densify_from_step <N>   First densification step (default: 500)\n");
    printf("  --densify_until_step <N>  Last densification step (default: 15000)\n");
    printf("  --densify_interval <N>    Densification interval (default: 100)\n");
    printf("  --opacity_reset_interval <N>  Reset opacity interval when densifying (default: 3000)\n");
    printf("  --eval_3d <0|1>       Use eval_3D rasterization during training (default: 0)\n");
    printf("  --parity_mode <0|1>   Use eval_3D parity replay mode; defaults to --eval_3d\n");
    printf("  --proper_ewa <0|1>    Use AAA proper EWA preprocessing during training (default: 0)\n");
    printf("  --training_preset <aaa|fast>  Hyperparameter defaults: aaa reference or old fast/L1 baseline (default: aaa)\n");
    printf("  --lambda_dssim <v>    DSSIM weight in combined loss, [0,1] (aaa default: 0.2; fast: 0)\n");
    printf("                         GPU DSSIM still requires GS3D_TRAIN_GPU_DSSIM_LOSS=1\n");
    printf("  --opacity_reg <v>     Opacity regularization weight (aaa default: 0.01; fast: 0)\n");
    printf("  --scale_reg <v>       Scale regularization weight (aaa default: 0.01; fast: 0)\n");
    printf("  --noise_lr <v>        Position noise LR (aaa default: 5e5; fast: 0)\n");
    printf("  --pos_lr_init <v>     Position LR init (default: 1.6e-4)\n");
    printf("  --pos_lr_final <v>    Position LR final (aaa default: 1.6e-6; fast: same as init)\n");
    printf("  --spatial_lr_scale <v>  Position LR spatial scale (default: 1)\n");
    printf("  --sh_degree_warmup <N>  Steps between SH degree increments (aaa default: 1000; fast: 0)\n");
}

// ---------------------------------------------------------------------------
// Training view: camera + ground truth image pair
// ---------------------------------------------------------------------------
struct TrainView {
    Camera cam;
    std::vector<float> gt_image;
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    TrainArgs args = parseVkTrainArgs(argc, argv);
    if (!args.parse_error.empty()) {
        printf("Error: %s\n", args.parse_error.c_str());
        printUsage(argv[0]);
        return 1;
    }
    if (!args.ply_path) {
        printf("Error: --ply is required\n");
        printUsage(argv[0]);
        return 1;
    }
    if (args.iterations <= 0) {
        printf("Error: --iterations must be > 0\n");
        return 1;
    }
    if (args.width <= 0 || args.height <= 0) {
        printf("Error: --width and --height must be > 0\n");
        return 1;
    }
    if (args.width > INT_MAX / args.height || args.width * args.height > INT_MAX / 3) {
        printf("Error: --width * --height is too large\n");
        return 1;
    }
    if (args.fov <= 0.0f || args.fov >= 180.0f) {
        printf("Error: --fov must be in the open range (0, 180)\n");
        return 1;
    }
    if (args.save_every < 0) {
        printf("Error: --save_every must be >= 0\n");
        return 1;
    }
    if (args.log_every <= 0) {
        printf("Error: --log_every must be > 0\n");
        return 1;
    }
    if (args.sh_degree < -1 || args.sh_degree > 3) {
        printf("Error: --sh_degree must be -1, 0, 1, 2, or 3\n");
        return 1;
    }
    if (args.lambda_dssim < 0.0f || args.lambda_dssim > 1.0f) {
        printf("Error: --lambda_dssim must be in [0, 1]\n");
        return 1;
    }
    if (args.opacity_reg < 0.0f || args.scale_reg < 0.0f || args.noise_lr < 0.0f) {
        printf("Error: --opacity_reg, --scale_reg, and --noise_lr must be >= 0\n");
        return 1;
    }
    if (args.pos_lr_init <= 0.0f || args.pos_lr_final <= 0.0f) {
        printf("Error: --pos_lr_init and --pos_lr_final must be > 0\n");
        return 1;
    }
    if (args.spatial_lr_scale <= 0.0f) {
        printf("Error: --spatial_lr_scale must be > 0\n");
        return 1;
    }
    if (args.sh_degree_warmup < 0) {
        printf("Error: --sh_degree_warmup must be >= 0\n");
        return 1;
    }

    // Init Vulkan
    VulkanContext ctx;
    if (!ctx.init()) {
        printf("Error: failed to initialize Vulkan\n");
        return 1;
    }
    printf("Vulkan device: %s\n", ctx.deviceName().c_str());

    // Load PLY model
    printf("Loading PLY: %s\n", args.ply_path);
    auto model = loadPly(args.ply_path);
    const int initial_N = model.data.count;
    const bool save_filter_3D = (model.data.filter_3D != nullptr) || args.eval_3D;
    printf("Loaded %d Gaussians, SH degree %d\n", model.data.count, model.data.sh_degree);

    RawGaussianParams raw = initRawFromGaussianData(model.data);

    const bool effective_parity_mode = args.parity_mode_provided ? args.parity_mode : args.eval_3D;

    // Render config
    RenderConfig cfg{};
    cfg.bg_color[0] = cfg.bg_color[1] = cfg.bg_color[2] = 0;
    cfg.scale_modifier = 1.0f;
    cfg.training = true;
    cfg.eval_3D = args.eval_3D;
    cfg.eval_3D_parity_mode = effective_parity_mode;
    cfg.sh_degree = (args.sh_degree >= 0) ? args.sh_degree : raw.sh_degree;

    if (args.cap_max_provided && args.cap_max > 0)
        args.densify = true;

    if (args.densify && args.densify_from_step <= 0) {
        printf("Error: --densify_from_step must be > 0 when densification is enabled\n");
        freeRawParams(raw);
        model.free();
        return 1;
    }
    if (args.densify && args.densify_interval <= 0) {
        printf("Error: --densify_interval must be > 0 when densification is enabled\n");
        freeRawParams(raw);
        model.free();
        return 1;
    }
    if (args.densify && args.densify_until_step < args.densify_from_step) {
        printf("Error: --densify_until_step must be >= --densify_from_step\n");
        freeRawParams(raw);
        model.free();
        return 1;
    }
    if (args.cap_max < 0) {
        printf("Error: --cap_max must be >= 0\n");
        freeRawParams(raw);
        model.free();
        return 1;
    }
    if (args.opacity_reset_interval < 0) {
        printf("Error: --opacity_reset_interval must be >= 0\n");
        freeRawParams(raw);
        model.free();
        return 1;
    }
    if (args.densify && !args.cap_max_provided)
        args.cap_max = std::max(initial_N + 1, static_cast<int>(initial_N * 1.5f));

    // Vulkan training config
    VkTrainingConfig tcfg{};
    tcfg.max_steps = args.iterations;
    tcfg.densify_from_step = args.densify ? args.densify_from_step : 0;
    tcfg.densify_until_step = args.densify_until_step;
    tcfg.densify_interval = args.densify_interval;
    tcfg.cap_max = args.densify ? args.cap_max : 0;
    tcfg.opacity_reset_interval = args.densify ? args.opacity_reset_interval : 0;
    tcfg.lambda_dssim = args.lambda_dssim;
    tcfg.opacity_reg = args.opacity_reg;
    tcfg.scale_reg = args.scale_reg;
    tcfg.pos_lr_init = args.pos_lr_init;
    tcfg.pos_lr_final = args.pos_lr_final;
    tcfg.spatial_lr_scale = args.spatial_lr_scale;
    tcfg.sh_degree_max = cfg.sh_degree;
    tcfg.sh_degree_warmup = args.sh_degree_warmup;
    tcfg.noise_lr = args.noise_lr;
    tcfg.eval_3D = args.eval_3D;
    tcfg.parity_mode = effective_parity_mode;
    tcfg.proper_ewa = args.proper_ewa;
    // Build training views
    std::vector<TrainView> views;

    if (args.cameras_json) {
        printf("Loading cameras from: %s\n", args.cameras_json);
        auto cam_entries = loadCamerasJson(args.cameras_json);
        printf("Found %zu cameras\n", cam_entries.size());

        for (auto& entry : cam_entries) {
            TrainView tv;
            tv.cam = entry.cam;

            bool loaded = false;
            if (args.gt_dir && !entry.img_name.empty()) {
                const char* exts[] = {".ppm", ".PPM", ""};
                for (auto ext : exts) {
                    std::string img_path = std::string(args.gt_dir) + "/" + entry.img_name + ext;
                    int iw, ih;
                    auto img = readPPM(img_path.c_str(), iw, ih);
                    if (!img.empty()) {
                        if (iw != tv.cam.width || ih != tv.cam.height) {
                            printf("  Resizing GT %s from %dx%d to %dx%d\n",
                                   img_path.c_str(), iw, ih, tv.cam.width, tv.cam.height);
                            std::vector<float> resized(tv.cam.width * tv.cam.height * 3);
                            for (int y = 0; y < tv.cam.height; y++) {
                                int sy = y * ih / tv.cam.height;
                                for (int x = 0; x < tv.cam.width; x++) {
                                    int sx = x * iw / tv.cam.width;
                                    for (int c = 0; c < 3; c++)
                                        resized[(y*tv.cam.width+x)*3+c] = img[(sy*iw+sx)*3+c];
                                }
                            }
                            img = std::move(resized);
                        }
                        // Convert HWC → CHW to match rasterizer output layout.
                        // Rasterizer writes out_image[ch * HW + px]; loss function
                        // compares rendered and GT linearly, so both must share layout.
                        const int npix = tv.cam.width * tv.cam.height;
                        std::vector<float> chw(npix * 3);
                        for (int px = 0; px < npix; px++)
                            for (int c = 0; c < 3; c++)
                                chw[c * npix + px] = img[px * 3 + c];
                        tv.gt_image = std::move(chw);
                        loaded = true;
                        break;
                    }
                }
            }
            if (!loaded) {
                printf("  Warning: no GT image found for '%s'\n", entry.img_name.c_str());
                if (args.require_all_gt) {
                    printf("Error: --require_all_gt is set and a camera is missing GT\n");
                    freeRawParams(raw);
                    model.free();
                    return 1;
                }
            }

            if (!tv.gt_image.empty())
                views.push_back(std::move(tv));
        }
    } else {
        TrainView tv;

        bool cam_specified = (args.cam_x != 0 || args.cam_y != 0 || args.cam_z != 0);
        if (cam_specified) {
            tv.cam = buildCamera(args.cam_x, args.cam_y, args.cam_z,
                                 args.fov, args.width, args.height,
                                 args.look_x, args.look_y, args.look_z);
        } else {
            tv.cam = autoCamera(model.data, args.width, args.height);
        }

        if (args.gt_path) {
            printf("Loading ground truth: %s\n", args.gt_path);
            int gt_w, gt_h;
            tv.gt_image = readPPM(args.gt_path, gt_w, gt_h);
            if (tv.gt_image.empty()) {
                printf("Error: failed to load PPM file: %s\n", args.gt_path);
                freeRawParams(raw);
                model.free();
                return 1;
            }
            if (gt_w != args.width || gt_h != args.height) {
                printf("Note: using GT image dimensions %dx%d\n", gt_w, gt_h);
                args.width = gt_w;
                args.height = gt_h;
                tv.cam.width = gt_w;
                tv.cam.height = gt_h;
            }
        } else {
            printf("No GT image provided. Using zero target (test mode).\n");
            int npix = tv.cam.width * tv.cam.height;
            tv.gt_image.resize(npix * 3, 0.0f);
        }

        views.push_back(std::move(tv));
    }

    if (views.empty()) {
        printf("Error: no training views with GT images available\n");
        freeRawParams(raw);
        model.free();
        return 1;
    }

    for (size_t i = 1; i < views.size(); ++i) {
        if (views[i].cam.width != views[0].cam.width || views[i].cam.height != views[0].cam.height) {
            printf("Error: mixed-resolution multi-view training is not supported yet (%dx%d vs %dx%d at view %zu)\n",
                   views[0].cam.width, views[0].cam.height,
                   views[i].cam.width, views[i].cam.height, i);
            freeRawParams(raw);
            model.free();
            return 1;
        }
    }

    std::vector<int> view_schedule;
    if (args.view_schedule_path) {
        try {
            view_schedule = loadViewSchedule(args.view_schedule_path);
        } catch (const std::exception& e) {
            printf("Error: %s\n", e.what());
            freeRawParams(raw);
            model.free();
            return 1;
        }
        if (static_cast<int>(view_schedule.size()) < args.iterations) {
            printf("Error: view schedule has %zu entries but --iterations is %d\n",
                   view_schedule.size(), args.iterations);
            freeRawParams(raw);
            model.free();
            return 1;
        }
        for (int iter = 0; iter < args.iterations; ++iter) {
            const int idx = view_schedule[static_cast<size_t>(iter)];
            if (idx < 0 || idx >= static_cast<int>(views.size())) {
                printf("Error: view schedule entry %d is out of range [0, %zu): %d\n",
                       iter, views.size(), idx);
                freeRawParams(raw);
                model.free();
                return 1;
            }
        }
    }

    // Create VulkanTrainer
    VulkanTrainer trainer(ctx, model.data, raw, cfg.sh_degree,
                          views[0].cam.width, views[0].cam.height, tcfg);

    // Raw data is now owned by the trainer internally; free our copy
    freeRawParams(raw);
    model.free();

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> view_dist(0, (int)views.size() - 1);

    printf("\n=== Vulkan Training ===\n");
    printf("  Device:      %s\n", ctx.deviceName().c_str());
    printf("  Gaussians:   %d\n", initial_N);
    printf("  SH degree:   %d\n", cfg.sh_degree);
    printf("  Iterations:  %d\n", args.iterations);
    printf("  Views:       %zu\n", views.size());
    printf("  Resolution:  %dx%d\n", views[0].cam.width, views[0].cam.height);
    printf("  view schedule: %s\n", view_schedule.empty() ? "random seed 42" : args.view_schedule_path);
    printf("  pos LR:      %.6f -> %.6f (scale=%.3f)\n", tcfg.pos_lr_init, tcfg.pos_lr_final, tcfg.spatial_lr_scale);
    printf("  SH warmup:   %d\n", tcfg.sh_degree_warmup);
    printf("  lambda_dssim:%.2f\n", tcfg.lambda_dssim);
    printf("  eval_3D:     %s\n", tcfg.eval_3D ? "ON" : "OFF");
    printf("  proper_ewa:  %s\n", tcfg.proper_ewa ? "ON" : "OFF");
    printf("  densify:     %s", tcfg.densify_from_step > 0 ? "ON" : "OFF");
    if (tcfg.densify_from_step > 0)
        printf(" (from=%d until=%d interval=%d cap_max=%d opacity_reset=%d)",
               tcfg.densify_from_step, tcfg.densify_until_step,
               tcfg.densify_interval, tcfg.cap_max,
               tcfg.opacity_reset_interval);
    printf("\n");
    printf("  save filter_3D: %s\n", save_filter_3D ? "ON" : "OFF");
    printf("  opacity_reg: %.4f\n", tcfg.opacity_reg);
    printf("  scale_reg:   %.4f\n", tcfg.scale_reg);
    printf("  noise_lr:    %.0f\n", tcfg.noise_lr);
    printf("\n");

    auto t_start = std::chrono::high_resolution_clock::now();
    float first_loss = 0, last_loss = 0;
    int nan_count = 0;
    int last_view_idx = 0;

    for (int iter = 0; iter < args.iterations; iter++) {
        int view_idx = !view_schedule.empty()
            ? view_schedule[static_cast<size_t>(iter)]
            : ((views.size() > 1) ? view_dist(rng) : 0);
        last_view_idx = view_idx;
        auto& tv = views[view_idx];

        if (iter == 0) {
            printf("  View %d view_matrix (col-major):\n", view_idx);
            for (int r = 0; r < 4; r++) {
                printf("    ");
                for (int c = 0; c < 4; c++)
                    printf(" %.6f", tv.cam.view_matrix[c*4+r]);
                printf("\n");
            }
            printf("  View %d viewproj_matrix (col-major):\n", view_idx);
            for (int r = 0; r < 4; r++) {
                printf("    ");
                for (int c = 0; c < 4; c++)
                    printf(" %.6f", tv.cam.viewproj_matrix[c*4+r]);
                printf("\n");
            }
        }

        // Get current raw params from trainer (may have been updated by densification)
        const RawGaussianParams& current_raw = trainer.raw_params();
        (void)current_raw;

        const bool is_final_iter = (iter == args.iterations - 1);
        // Match CUDA reference training: final forward/backward runs, final Adam update does not.
        float loss = trainer.step(tv.cam, cfg, tv.gt_image.data(),
                                  tv.cam.width, tv.cam.height,
                                  !is_final_iter);

        if (std::isnan(loss) || std::isinf(loss)) {
            nan_count++;
            if (nan_count <= 3)
                printf("  Warning: %s loss at iter %d\n",
                       std::isnan(loss) ? "NaN" : "Inf", iter);
            loss = 0.0f;
        }

        if (iter == 0) {
            first_loss = loss;
            // Save first-iteration rendered image for debugging.
            const float* rendered = trainer.rendered_image();
            if (rendered) {
                int npix1 = tv.cam.width * tv.cam.height;
                std::vector<float> hwc1(npix1 * 3);
                for (int px = 0; px < npix1; px++)
                    for (int c = 0; c < 3; c++)
                        hwc1[px * 3 + c] = rendered[c * npix1 + px];
                char render_path[512];
                snprintf(render_path, sizeof(render_path), "%s.iter1.render.ppm", args.output_path);
                writePPM(render_path, hwc1.data(), tv.cam.width, tv.cam.height);
                printf("  Saved iter-1 render: %s (loss=%.6f, view=%d)\n",
                       render_path, loss, view_idx);
            }
        }
        last_loss = loss;

        if (iter == 0 || (iter + 1) % args.log_every == 0 || iter == args.iterations - 1) {
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            double iter_per_sec = (iter + 1) / elapsed;
            if (views.size() > 1)
                printf("iter %5d/%d  loss=%.6f  view=%d  (%.1f it/s)\n",
                       iter + 1, args.iterations, loss, view_idx, iter_per_sec);
            else
                printf("iter %5d/%d  loss=%.6f  (%.1f it/s)\n",
                       iter + 1, args.iterations, loss, iter_per_sec);
        }

        if (args.save_every > 0 && (iter + 1) % args.save_every == 0) {
            char ckpt_path[512];
            snprintf(ckpt_path, sizeof(ckpt_path), "%s.iter%d.ply", args.output_path, iter + 1);
            savePly(ckpt_path, trainer.raw_params(),
                    save_filter_3D ? trainer.filter_3D().data() : nullptr);
            printf("  Saved checkpoint: %s\n", ckpt_path);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    printf("\nTraining complete.\n");
    printf("  Total time:   %.1f s (%.1f it/s)\n", total_sec, args.iterations / total_sec);
    printf("  First loss:   %.6f\n", first_loss);
    printf("  Final loss:   %.6f\n", last_loss);
    if (nan_count > 0)
        printf("  NaN/Inf steps: %d\n", nan_count);
    if (first_loss > 0)
        printf("  Loss ratio:   %.4f (%.1f%% reduction)\n",
               last_loss / first_loss, (1.0f - last_loss / first_loss) * 100.0f);

    // Save final rendered image (convert CHW→HWC for writePPM)
    const float* rendered = trainer.rendered_image();
    if (rendered) {
        const auto& final_view = views[last_view_idx];
        int pw = final_view.cam.width, ph = final_view.cam.height;
        int npix = pw * ph;
        std::vector<float> hwc(npix * 3);
        for (int px = 0; px < npix; px++)
            for (int c = 0; c < 3; c++)
                hwc[px * 3 + c] = rendered[c * npix + px];
        char render_path[512];
        snprintf(render_path, sizeof(render_path), "%s.render.ppm", args.output_path);
        writePPM(render_path, hwc.data(), pw, ph);
        printf("  Saved final render: %s\n", render_path);
    }

    // Save trained model
    printf("Saving trained model: %s\n", args.output_path);
    if (!savePly(args.output_path, trainer.raw_params(),
                 save_filter_3D ? trainer.filter_3D().data() : nullptr)) {
        printf("Error: failed to save PLY: %s\n", args.output_path);
        return 1;
    }
    printf("Done.\n");
    std::fflush(nullptr);
    std::_Exit(0);
}
