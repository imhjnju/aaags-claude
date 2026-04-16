// Auto-generated from rasterize_backward.cl -- do not edit
static const char rasterize_backward_cl_src[] = R"CL(
// rasterize_backward.cl -- Backward pass for tile-based alpha blending
// Standard 2D conic path only.

#define TILE_W 16
#define TILE_H 16
#define BLOCK_SIZE 256

inline void atomicAdd_f(__global volatile float* addr, float val) {
    union { float f; unsigned int u; } old_v, new_v;
    do {
        old_v.f = *addr;
        new_v.f = old_v.f + val;
    } while (atomic_cmpxchg((__global volatile unsigned int*)addr,
                              old_v.u, new_v.u) != old_v.u);
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void rasterize_backward(
    __global const float* restrict means2D,
    __global const float* restrict conics,
    __global const float* restrict rgb,
    __global const float* restrict opacities_2d,
    __global const uint*  restrict values_sorted,
    __global const uint*  restrict tile_ranges,
    const int width,
    const int height,
    const float bg_r,
    const float bg_g,
    const float bg_b,
    const int grid_x,
    __global const float* restrict d_image,
    __global const float* restrict T_final,
    __global const int*   restrict n_contrib,
    __global float* restrict d_means2D,
    __global float* restrict d_conics,
    __global float* restrict d_rgb_out,
    __global float* restrict d_opacities_2d
)
{
    int tile_id = get_group_id(0);
    int local_id = get_local_id(0);

    int tx = tile_id % grid_x;
    int ty = tile_id / grid_x;
    int px = tx * TILE_W + (local_id & 15);
    int py = ty * TILE_H + (local_id >> 4);

    uint range_start = tile_ranges[tile_id * 2];
    uint range_end   = tile_ranges[tile_id * 2 + 1];

    __local float s_xy_x[BLOCK_SIZE];
    __local float s_xy_y[BLOCK_SIZE];
    __local float s_conic_a[BLOCK_SIZE];
    __local float s_conic_b[BLOCK_SIZE];
    __local float s_conic_c[BLOCK_SIZE];
    __local float s_rgb_r[BLOCK_SIZE];
    __local float s_rgb_g[BLOCK_SIZE];
    __local float s_rgb_b[BLOCK_SIZE];
    __local float s_opa[BLOCK_SIZE];
    __local uint  s_gid[BLOCK_SIZE];

    bool valid_pixel = (px < width && py < height);
    int pix = valid_pixel ? (py * width + px) : 0;

    #define MAX_CONTRIB 128
    uint  c_gid[MAX_CONTRIB];
    float c_alpha[MAX_CONTRIB];
    int   c_count = 0;

    float T_replay = 1.0f;
    bool done = !valid_pixel;

    for (uint batch_start = range_start; batch_start < range_end; batch_start += BLOCK_SIZE) {
        uint load_idx = batch_start + (uint)local_id;
        if (load_idx < range_end) {
            uint gid = values_sorted[load_idx];
            s_xy_x[local_id]    = means2D[gid * 2];
            s_xy_y[local_id]    = means2D[gid * 2 + 1];
            s_conic_a[local_id] = conics[gid * 3];
            s_conic_b[local_id] = conics[gid * 3 + 1];
            s_conic_c[local_id] = conics[gid * 3 + 2];
            s_rgb_r[local_id]   = rgb[gid * 3];
            s_rgb_g[local_id]   = rgb[gid * 3 + 1];
            s_rgb_b[local_id]   = rgb[gid * 3 + 2];
            s_opa[local_id]     = opacities_2d[gid];
            s_gid[local_id]     = gid;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (!done) {
            uint count = min((uint)BLOCK_SIZE, range_end - batch_start);
            for (uint j = 0; j < count; j++) {
                float dx = s_xy_x[j] - (float)px;
                float dy = s_xy_y[j] - (float)py;
                float power = -0.5f * (s_conic_a[j]*dx*dx + s_conic_c[j]*dy*dy) - s_conic_b[j]*dx*dy;
                if (power > 0.0f) continue;

                float alpha = fmin(0.99f, s_opa[j] * exp(power));
                if (alpha < (1.0f / 255.0f)) continue;

                float test_T = T_replay * (1.0f - alpha);
                if (test_T < 0.0001f) {
                    done = true;
                    break;
                }

                if (c_count < MAX_CONTRIB) {
                    c_gid[c_count]   = s_gid[j];
                    c_alpha[c_count] = alpha;
                    c_count++;
                }
                T_replay = test_T;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (valid_pixel && c_count > 0) {
        float d_C[3] = { d_image[pix*3], d_image[pix*3+1], d_image[pix*3+2] };

        float T_accum = T_replay;
        float C_accum[3] = {
            bg_r * T_replay,
            bg_g * T_replay,
            bg_b * T_replay
        };

        for (int i = c_count - 1; i >= 0; i--) {
            uint gauss_idx = c_gid[i];
            float alpha = c_alpha[i];

            float one_minus_alpha = 1.0f - alpha;
            float T_i = (one_minus_alpha > 1e-6f) ? T_accum / one_minus_alpha : 0.0f;

            float dx = means2D[gauss_idx*2]   - (float)px;
            float dy = means2D[gauss_idx*2+1] - (float)py;
            float con_a = conics[gauss_idx*3];
            float con_b = conics[gauss_idx*3+1];
            float con_c = conics[gauss_idx*3+2];

            float w_rgb = alpha * T_i;
            atomicAdd_f(&d_rgb_out[gauss_idx*3],   w_rgb * d_C[0]);
            atomicAdd_f(&d_rgb_out[gauss_idx*3+1], w_rgb * d_C[1]);
            atomicAdd_f(&d_rgb_out[gauss_idx*3+2], w_rgb * d_C[2]);

            float d_alpha = 0.0f;
            d_alpha += (T_i * rgb[gauss_idx*3]   - C_accum[0]) * d_C[0];
            d_alpha += (T_i * rgb[gauss_idx*3+1] - C_accum[1]) * d_C[1];
            d_alpha += (T_i * rgb[gauss_idx*3+2] - C_accum[2]) * d_C[2];

            C_accum[0] = rgb[gauss_idx*3]   * alpha * T_i + C_accum[0];
            C_accum[1] = rgb[gauss_idx*3+1] * alpha * T_i + C_accum[1];
            C_accum[2] = rgb[gauss_idx*3+2] * alpha * T_i + C_accum[2];
            T_accum = T_i;

            float power = -0.5f*(con_a*dx*dx + con_c*dy*dy) - con_b*dx*dy;
            float exp_power = exp(power);
            float d_power, d_opa;
            if (alpha < 0.99f) {
                d_power = d_alpha * opacities_2d[gauss_idx] * exp_power;
                d_opa = d_alpha * exp_power;
            } else {
                d_power = 0.0f;
                d_opa = 0.0f;
            }

            atomicAdd_f(&d_opacities_2d[gauss_idx], d_opa);

            atomicAdd_f(&d_means2D[gauss_idx*2],   d_power * (-con_a * dx - con_b * dy));
            atomicAdd_f(&d_means2D[gauss_idx*2+1], d_power * (-con_c * dy - con_b * dx));

            atomicAdd_f(&d_conics[gauss_idx*3],   d_power * (-0.5f * dx * dx));
            atomicAdd_f(&d_conics[gauss_idx*3+1], d_power * (-dx * dy));
            atomicAdd_f(&d_conics[gauss_idx*3+2], d_power * (-0.5f * dy * dy));
        }
    }
}
)CL";
