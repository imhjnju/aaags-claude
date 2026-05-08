#include "camera_utils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cfloat>

// Build a look-at view matrix (column-major, matching GLM/CUDA convention)
// eye: camera position, center: look-at target, up: world up vector
void buildLookAt(const float eye[3], const float center[3], const float up[3],
                 float view_matrix[16]) {
    // Forward = normalize(center - eye)
    float f[3] = {center[0]-eye[0], center[1]-eye[1], center[2]-eye[2]};
    float flen = sqrtf(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
    if (flen < 1e-8f) {  // degenerate: eye == center
        std::fill(view_matrix, view_matrix + 16, 0.0f);
        view_matrix[0] = view_matrix[5] = view_matrix[10] = view_matrix[15] = 1.0f;
        return;
    }
    f[0]/=flen; f[1]/=flen; f[2]/=flen;

    // Right = normalize(forward x up)
    float r[3] = {
        f[1]*up[2] - f[2]*up[1],
        f[2]*up[0] - f[0]*up[2],
        f[0]*up[1] - f[1]*up[0]
    };
    float rlen = sqrtf(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    if (rlen < 1e-8f) {  // degenerate: forward parallel to up
        std::fill(view_matrix, view_matrix + 16, 0.0f);
        view_matrix[0] = view_matrix[5] = view_matrix[10] = view_matrix[15] = 1.0f;
        return;
    }
    r[0]/=rlen; r[1]/=rlen; r[2]/=rlen;

    // True up = right x forward
    float u[3] = {
        r[1]*f[2] - r[2]*f[1],
        r[2]*f[0] - r[0]*f[2],
        r[0]*f[1] - r[1]*f[0]
    };

    // Column-major view matrix: R^T | -R^T * eye
    // Column 0: right
    view_matrix[0]  = r[0];  view_matrix[1]  = u[0];  view_matrix[2]  = f[0]; view_matrix[3]  = 0;
    // Column 1: up
    view_matrix[4]  = r[1];  view_matrix[5]  = u[1];  view_matrix[6]  = f[1]; view_matrix[7]  = 0;
    // Column 2: forward
    view_matrix[8]  = r[2];  view_matrix[9]  = u[2];  view_matrix[10] = f[2]; view_matrix[11] = 0;
    // Column 3: translation = -R^T * eye
    view_matrix[12] = -(r[0]*eye[0] + r[1]*eye[1] + r[2]*eye[2]);
    view_matrix[13] = -(u[0]*eye[0] + u[1]*eye[1] + u[2]*eye[2]);
    view_matrix[14] = -(f[0]*eye[0] + f[1]*eye[1] + f[2]*eye[2]);
    view_matrix[15] = 1;
}

// Build perspective projection matrix (column-major)
void buildPerspective(float tan_fovx, float tan_fovy, float znear, float zfar,
                      float proj[16]) {
    memset(proj, 0, 16 * sizeof(float));
    proj[0]  = 1.0f / tan_fovx;
    proj[5]  = 1.0f / tan_fovy;
    proj[10] = zfar / (zfar - znear);
    proj[11] = 1.0f;
    proj[14] = -(zfar * znear) / (zfar - znear);
}

// Multiply two 4x4 column-major matrices: out = A * B
void mat4Mul(const float A[16], const float B[16], float out[16]) {
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++) {
            float sum = 0;
            for (int k = 0; k < 4; k++)
                sum += A[k*4 + row] * B[col*4 + k];
            out[col*4 + row] = sum;
        }
}

// Auto-compute camera from scene bounding box
Camera autoCamera(const GaussianData& g, int width, int height) {
    // Compute scene center and extent (using percentiles for robustness)
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
    float cx = (xs[p5] + xs[p95]) * 0.5f;
    float cy = (ys[p5] + ys[p95]) * 0.5f;
    float cz = (zs[p5] + zs[p95]) * 0.5f;
    float span_x = xs[p95] - xs[p5];
    float span_y = ys[p95] - ys[p5];
    float span_z = zs[p95] - zs[p5];
    float scene_radius = sqrtf(span_x*span_x + span_y*span_y + span_z*span_z) * 0.5f;
    if (scene_radius < 1e-6f) scene_radius = 1.0f;  // fallback for tiny scenes

    printf("Scene center: (%.3f, %.3f, %.3f), radius: %.3f\n", cx, cy, cz, scene_radius);

    Camera cam{};
    cam.width = width;
    cam.height = height;
    float fov_deg = 50.0f;
    cam.tan_fovx = tanf(fov_deg * 0.5f * 3.14159265f / 180.0f);
    cam.tan_fovy = cam.tan_fovx * (float)height / (float)width;

    // Place camera at a distance to see the whole scene
    float dist = scene_radius / cam.tan_fovx * 0.8f;  // tighter framing
    // Camera position: offset from center along a diagonal direction
    float eye[3] = {cx + dist * 0.6f, cy + dist * 0.3f, cz - dist * 0.7f};
    float center[3] = {cx, cy, cz};
    float up[3] = {0, 1, 0};  // Y-up for correct orientation

    cam.cam_pos[0] = eye[0];
    cam.cam_pos[1] = eye[1];
    cam.cam_pos[2] = eye[2];

    buildLookAt(eye, center, up, cam.view_matrix);

    // Build viewproj = view * proj
    float proj[16];
    buildPerspective(cam.tan_fovx, cam.tan_fovy, 0.01f, 1000.0f, proj);
    // viewproj = proj * view (projection applied after view transform)
    // mat4Mul(A, B) computes A*B for column-major matrices
    // Original Python: full_proj = world_view_cm.bmm(projection_cm) = column-major(Proj * View)
    mat4Mul(proj, cam.view_matrix, cam.viewproj_matrix);

    printf("Camera eye: (%.3f, %.3f, %.3f), dist: %.3f\n", eye[0], eye[1], eye[2], dist);
    return cam;
}

// Load camera from cameras.json (3DGS format)
// cameras.json stores: position (world camera center), rotation (W2C 3x3), fx, fy, width, height
Camera loadCameraJson(const char* json_path, int cam_id) {
    std::ifstream jf(json_path);
    if (!jf.is_open())
        throw std::runtime_error(std::string("Cannot open ") + json_path);

    // Minimal JSON parser: find the camera with matching id
    std::string content((std::istreambuf_iterator<char>(jf)),
                         std::istreambuf_iterator<char>());

    // Parse all cameras - simple approach: find "id": cam_id
    // Use a quick and dirty parser for the specific structure
    Camera cam{};

    // Extract arrays from JSON using string scanning
    auto findNum = [&](const std::string& s, size_t pos) -> std::pair<double, size_t> {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == ',' || s[pos] == '\n' || s[pos] == '\r'))
            pos++;
        size_t end = pos;
        while (end < s.size() && (s[end] == '-' || s[end] == '.' || s[end] == 'e' || s[end] == 'E'
               || s[end] == '+' || (s[end] >= '0' && s[end] <= '9')))
            end++;
        return {std::stod(s.substr(pos, end - pos)), end};
    };

    // Find camera by id
    std::string id_pattern = "\"id\": " + std::to_string(cam_id);
    size_t cam_pos = content.find(id_pattern);
    if (cam_pos == std::string::npos)
        throw std::runtime_error("Camera id " + std::to_string(cam_id) + " not found");

    // Find the enclosing object (scan back to '{')
    size_t obj_start = content.rfind('{', cam_pos);
    size_t obj_end = content.find('}', cam_pos);
    // Find closing '}' that matches - need to handle nested
    int depth = 1;
    size_t p = obj_start + 1;
    while (p < content.size() && depth > 0) {
        if (content[p] == '{') depth++;
        else if (content[p] == '}') depth--;
        p++;
    }
    obj_end = p;
    std::string obj = content.substr(obj_start, obj_end - obj_start);

    // Parse width, height
    auto parseIntField = [&](const std::string& field) -> int {
        size_t pos = obj.find("\"" + field + "\"");
        pos = obj.find(':', pos) + 1;
        return (int)findNum(obj, pos).first;
    };
    auto parseFloatField = [&](const std::string& field) -> float {
        size_t pos = obj.find("\"" + field + "\"");
        pos = obj.find(':', pos) + 1;
        return (float)findNum(obj, pos).first;
    };

    cam.width = parseIntField("width");
    cam.height = parseIntField("height");
    float fx = parseFloatField("fx");
    float fy = parseFloatField("fy");
    cam.tan_fovx = (float)cam.width / (2.0f * fx);
    cam.tan_fovy = (float)cam.height / (2.0f * fy);

    // Parse position [x, y, z]
    size_t pos_start = obj.find("\"position\"");
    pos_start = obj.find('[', pos_start) + 1;
    auto [px, p1] = findNum(obj, pos_start);
    auto [py, p2] = findNum(obj, p1);
    auto [pz, p3] = findNum(obj, p2);
    cam.cam_pos[0] = (float)px;
    cam.cam_pos[1] = (float)py;
    cam.cam_pos[2] = (float)pz;

    // Parse rotation [[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]]
    // This is the W2C rotation matrix R
    float R[3][3];
    size_t rot_start = obj.find("\"rotation\"");
    rot_start = obj.find('[', rot_start) + 1; // outer [
    for (int row = 0; row < 3; row++) {
        rot_start = obj.find('[', rot_start) + 1; // inner [
        for (int col = 0; col < 3; col++) {
            auto [val, next] = findNum(obj, rot_start);
            R[row][col] = (float)val;
            rot_start = next;
        }
        rot_start = obj.find(']', rot_start) + 1;
    }

    // cameras.json stores C2W (camera-to-world) rotation, NOT W2C!
    // (Python code misleadingly names the variable 'W2C' but it's actually inv(W2C) = C2W)
    // W2C rotation = R_c2w^T, W2C translation = -R_c2w^T * cam_center
    // view_matrix[col*4+row] = R_w2c(row,col) = R_c2w(col,row) = R[col][row]
    float t[3] = {
        -(R[0][0]*cam.cam_pos[0] + R[1][0]*cam.cam_pos[1] + R[2][0]*cam.cam_pos[2]),
        -(R[0][1]*cam.cam_pos[0] + R[1][1]*cam.cam_pos[1] + R[2][1]*cam.cam_pos[2]),
        -(R[0][2]*cam.cam_pos[0] + R[1][2]*cam.cam_pos[1] + R[2][2]*cam.cam_pos[2])
    };

    // Store W2C = R_c2w^T as column-major: view_matrix[col*4+row] = R[col][row]
    // Column 0
    cam.view_matrix[0] = R[0][0]; cam.view_matrix[1] = R[0][1]; cam.view_matrix[2] = R[0][2]; cam.view_matrix[3] = 0;
    // Column 1
    cam.view_matrix[4] = R[1][0]; cam.view_matrix[5] = R[1][1]; cam.view_matrix[6] = R[1][2]; cam.view_matrix[7] = 0;
    // Column 2
    cam.view_matrix[8] = R[2][0]; cam.view_matrix[9] = R[2][1]; cam.view_matrix[10] = R[2][2]; cam.view_matrix[11] = 0;
    // Column 3
    cam.view_matrix[12] = t[0]; cam.view_matrix[13] = t[1]; cam.view_matrix[14] = t[2]; cam.view_matrix[15] = 1;

    // Build viewproj = proj * view
    float proj[16];
    buildPerspective(cam.tan_fovx, cam.tan_fovy, 0.01f, 100.0f, proj);
    mat4Mul(proj, cam.view_matrix, cam.viewproj_matrix);

    printf("Camera %d: %dx%d, fx=%.1f, fy=%.1f, pos=(%.3f,%.3f,%.3f)\n",
           cam_id, cam.width, cam.height, fx, fy,
           cam.cam_pos[0], cam.cam_pos[1], cam.cam_pos[2]);

    return cam;
}
