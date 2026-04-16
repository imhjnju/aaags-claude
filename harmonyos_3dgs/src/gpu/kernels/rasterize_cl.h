// Auto-generated from rasterize.cl -- do not edit
static const char rasterize_cl_src[] = R"CL(
// rasterize.cl -- Tile-based alpha blending kernel for 3DGS
// Supports both standard 2D conic and AAA-Gaussians 3D evaluation.
// One work-group per tile, one work-item per pixel within the tile.

#define TILE_W 16
#define TILE_H 16
#define BLOCK_SIZE 256

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void rasterize(
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
    __global float* restrict out_image,
    // AAA-Gaussians additions
    const int eval_3D,
    __global const float* restrict gauss2screen,  // [N*16] row-major, or NULL
    __global const float* restrict depths,         // [N] view-space z (fallback)
    // StopThePop per-pixel depth
    __global const float* restrict cov3D_inv,      // [N*6] inverse 3D covariance
    __global const float* restrict mean_offset,    // [N*3] pos - cam_pos
    __constant const float* restrict inverse_vp,   // [16] inverse viewproj
    __constant const float* restrict cam_pos       // [3] camera position
)
{
    int tile_id = get_group_id(0);
    int local_id = get_local_id(0);

    int tx = tile_id % grid_x;
    int ty = tile_id / grid_x;
    int px = tx * TILE_W + (local_id & 15);
    int py = ty * TILE_H + (local_id >> 4);

    uint range_start = tile_ranges[tile_id * 2];
    uint range_end = tile_ranges[tile_id * 2 + 1];

    // Shared memory for batch loading
    __local float s_xy_x[BLOCK_SIZE];
    __local float s_xy_y[BLOCK_SIZE];
    __local float s_conic_a[BLOCK_SIZE];
    __local float s_conic_b[BLOCK_SIZE];
    __local float s_conic_c[BLOCK_SIZE];
    __local float s_rgb_r[BLOCK_SIZE];
    __local float s_rgb_g[BLOCK_SIZE];
    __local float s_rgb_b[BLOCK_SIZE];
    __local float s_opa[BLOCK_SIZE];

    // For eval_3D: gauss2screen matrix rows (16 floats per Gaussian)
    // We load 4 rows of 4 floats each into shared memory
    __local float s_g2s_r0[BLOCK_SIZE * 4];  // row0 [4 floats each]
    __local float s_g2s_r1[BLOCK_SIZE * 4];  // row1
    __local float s_g2s_r3[BLOCK_SIZE * 4];  // row3 (for perspective)
    __local float s_depth[BLOCK_SIZE];        // view-space depth (fallback)
    // StopThePop: per-pixel depth via depthAlongRay
    __local float s_cov3d_inv[BLOCK_SIZE * 6];  // [6 floats per Gaussian]
    __local float s_mean_off[BLOCK_SIZE * 3];   // [3 floats per Gaussian]

    __local int s_wg_done;

    float T = 1.0f;
    float C_r = 0.0f, C_g = 0.0f, C_b = 0.0f;
    bool done = (px >= width || py >= height);

    float fpx = (float)px + 0.5f;
    float fpy = (float)py + 0.5f;

    // StopThePop: compute per-pixel ray direction (once per pixel)
    float ray_dir[3] = {0, 0, 1};
    if (eval_3D && inverse_vp != 0 && px < width && py < height) {
        float ndc_x = (2.0f * fpx) / (float)width - 1.0f;
        float ndc_y = (2.0f * fpy) / (float)height - 1.0f;
        float p4[4] = {ndc_x, ndc_y, 0.0f, 1.0f};
        float wp[4];
        for (int i = 0; i < 4; i++)
            wp[i] = inverse_vp[i]*p4[0] + inverse_vp[4+i]*p4[1] + inverse_vp[8+i]*p4[2] + inverse_vp[12+i]*p4[3];
        float wi = 1.0f / (wp[3] + 1e-10f);
        ray_dir[0] = wp[0]*wi - cam_pos[0];
        ray_dir[1] = wp[1]*wi - cam_pos[1];
        ray_dir[2] = wp[2]*wi - cam_pos[2];
        float rl = sqrt(ray_dir[0]*ray_dir[0] + ray_dir[1]*ray_dir[1] + ray_dir[2]*ray_dir[2]);
        if (rl > 1e-10f) { ray_dir[0]/=rl; ray_dir[1]/=rl; ray_dir[2]/=rl; }
    }

    // kBuffer for per-pixel sorting (eval_3D only, K=16)
    #define KBUF_K 16
    float kb_depth[KBUF_K];
    float kb_alpha[KBUF_K];
    float kb_r[KBUF_K], kb_g[KBUF_K], kb_b[KBUF_K];
    int kb_count = 0;

    for (uint batch_start = range_start; batch_start < range_end; batch_start += BLOCK_SIZE) {
        uint load_idx = batch_start + (uint)local_id;
        if (load_idx < range_end) {
            uint gid = values_sorted[load_idx];
            s_rgb_r[local_id] = rgb[gid * 3];
            s_rgb_g[local_id] = rgb[gid * 3 + 1];
            s_rgb_b[local_id] = rgb[gid * 3 + 2];
            s_opa[local_id] = opacities_2d[gid];

            if (eval_3D) {
                // Load gauss2screen rows 0, 1, 3 (row 2 not needed for per-pixel eval)
                int base = gid * 16;
                s_g2s_r0[local_id*4]   = gauss2screen[base + 0];
                s_g2s_r0[local_id*4+1] = gauss2screen[base + 1];
                s_g2s_r0[local_id*4+2] = gauss2screen[base + 2];
                s_g2s_r0[local_id*4+3] = gauss2screen[base + 3];
                s_g2s_r1[local_id*4]   = gauss2screen[base + 4];
                s_g2s_r1[local_id*4+1] = gauss2screen[base + 5];
                s_g2s_r1[local_id*4+2] = gauss2screen[base + 6];
                s_g2s_r1[local_id*4+3] = gauss2screen[base + 7];
                s_g2s_r3[local_id*4]   = gauss2screen[base + 12];
                s_g2s_r3[local_id*4+1] = gauss2screen[base + 13];
                s_g2s_r3[local_id*4+2] = gauss2screen[base + 14];
                s_g2s_r3[local_id*4+3] = gauss2screen[base + 15];
                s_depth[local_id] = depths[gid];
                // StopThePop: load cov3D_inv and mean_offset
                if (cov3D_inv != 0) {
                    for (int ci = 0; ci < 6; ci++)
                        s_cov3d_inv[local_id*6+ci] = cov3D_inv[gid*6+ci];
                    for (int ci = 0; ci < 3; ci++)
                        s_mean_off[local_id*3+ci] = mean_offset[gid*3+ci];
                }
            } else {
                s_xy_x[local_id] = means2D[gid * 2];
                s_xy_y[local_id] = means2D[gid * 2 + 1];
                s_conic_a[local_id] = conics[gid * 3];
                s_conic_b[local_id] = conics[gid * 3 + 1];
                s_conic_c[local_id] = conics[gid * 3 + 2];
            }
        }

        if (local_id == 0) s_wg_done = 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (!done) {
            uint count = min((uint)BLOCK_SIZE, range_end - batch_start);
            for (uint j = 0; j < count; j++) {
                float power;
                if (eval_3D) {
                    // 3D plane-based evaluation
                    int b = j * 4;
                    float px0 = s_g2s_r0[b] - s_g2s_r3[b] * fpx;
                    float px1 = s_g2s_r0[b+1] - s_g2s_r3[b+1] * fpx;
                    float px2 = s_g2s_r0[b+2] - s_g2s_r3[b+2] * fpx;
                    float px3 = s_g2s_r0[b+3] - s_g2s_r3[b+3] * fpx;
                    float py0 = s_g2s_r1[b] - s_g2s_r3[b] * fpy;
                    float py1 = s_g2s_r1[b+1] - s_g2s_r3[b+1] * fpy;
                    float py2 = s_g2s_r1[b+2] - s_g2s_r3[b+2] * fpy;
                    float py3 = s_g2s_r1[b+3] - s_g2s_r3[b+3] * fpy;

                    float dx = px1*py2 - px2*py1;
                    float dy = px2*py0 - px0*py2;
                    float dz = px0*py1 - px1*py0;
                    float mx = px3*py0 - px0*py3;
                    float my = px3*py1 - px1*py3;
                    float mz = px3*py2 - px2*py3;
                    float dd = dx*dx + dy*dy + dz*dz;
                    float contrib = (mx*mx + my*my + mz*mz) / dd;
                    power = -0.5f * contrib;
                    if (power > 0.0f) continue;

                    float alpha = fmin(0.99f, s_opa[j] * native_exp(power));
                    if (alpha < (1.0f / 255.0f)) continue;

                    // StopThePop: per-pixel depth via depthAlongRay
                    float pix_depth = s_depth[j];  // fallback: view-space z
                    if (cov3D_inv != 0) {
                        int ci_base = j * 6;
                        int mo_base = j * 3;
                        float ci0=s_cov3d_inv[ci_base], ci1=s_cov3d_inv[ci_base+1], ci2=s_cov3d_inv[ci_base+2];
                        float ci3=s_cov3d_inv[ci_base+3], ci4=s_cov3d_inv[ci_base+4], ci5=s_cov3d_inv[ci_base+5];
                        float mo0=s_mean_off[mo_base], mo1=s_mean_off[mo_base+1], mo2=s_mean_off[mo_base+2];
                        // Sigma_inv * ray_dir
                        float sv0 = ci0*ray_dir[0] + ci1*ray_dir[1] + ci2*ray_dir[2];
                        float sv1 = ci1*ray_dir[0] + ci3*ray_dir[1] + ci4*ray_dir[2];
                        float sv2 = ci2*ray_dir[0] + ci4*ray_dir[1] + ci5*ray_dir[2];
                        float num = mo0*sv0 + mo1*sv1 + mo2*sv2;
                        float den = ray_dir[0]*sv0 + ray_dir[1]*sv1 + ray_dir[2]*sv2;
                        if (fabs(den) > 1e-10f) {
                            float ptd = num / den;
                            if (ptd > 0.0f) pix_depth = ptd;
                        }
                    }

                    // If buffer full, pop front (closest) and blend
                    if (kb_count >= KBUF_K) {
                        float fa = kb_alpha[0];
                        float test_T = T * (1.0f - fa);
                        if (test_T < 0.0001f) { done = true; break; }
                        float w = fa * T;
                        C_r = mad(kb_r[0], w, C_r);
                        C_g = mad(kb_g[0], w, C_g);
                        C_b = mad(kb_b[0], w, C_b);
                        T = test_T;
                        for (int k = 0; k < KBUF_K - 1; k++) {
                            kb_depth[k] = kb_depth[k+1]; kb_alpha[k] = kb_alpha[k+1];
                            kb_r[k] = kb_r[k+1]; kb_g[k] = kb_g[k+1]; kb_b[k] = kb_b[k+1];
                        }
                        kb_count = KBUF_K - 1;
                    }
                    // Sorted insertion
                    int pos = kb_count;
                    for (int k = kb_count - 1; k >= 0; k--) {
                        if (kb_depth[k] > pix_depth) pos = k; else break;
                    }
                    for (int k = kb_count; k > pos; k--) {
                        kb_depth[k] = kb_depth[k-1]; kb_alpha[k] = kb_alpha[k-1];
                        kb_r[k] = kb_r[k-1]; kb_g[k] = kb_g[k-1]; kb_b[k] = kb_b[k-1];
                    }
                    kb_depth[pos] = pix_depth; kb_alpha[pos] = alpha;
                    kb_r[pos] = s_rgb_r[j]; kb_g[pos] = s_rgb_g[j]; kb_b[pos] = s_rgb_b[j];
                    kb_count++;
                    continue;  // skip the direct blend below
                } else {
                    float dx = s_xy_x[j] - (float)px;
                    float dy = s_xy_y[j] - (float)py;
                    power = mad(-0.5f, mad(s_conic_a[j], dx*dx, s_conic_c[j]*dy*dy),
                                -s_conic_b[j]*dx*dy);
                    if (power > 0.0f) continue;
                }

                float alpha = fmin(0.99f, s_opa[j] * native_exp(power));
                if (alpha < (1.0f / 255.0f)) continue;

                float test_T = T * (1.0f - alpha);
                if (test_T < 0.0001f) {
                    done = true;
                    break;
                }

                float w = alpha * T;
                C_r = mad(s_rgb_r[j], w, C_r);
                C_g = mad(s_rgb_g[j], w, C_g);
                C_b = mad(s_rgb_b[j], w, C_b);
                T = test_T;
            }
        }

        if (!done) s_wg_done = 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (s_wg_done) break;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Flush remaining kBuffer entries (eval_3D only)
    if (eval_3D) {
        for (int k = 0; k < kb_count; k++) {
            float fa = kb_alpha[k];
            float test_T = T * (1.0f - fa);
            if (test_T < 0.0001f) break;
            float w = fa * T;
            C_r = mad(kb_r[k], w, C_r);
            C_g = mad(kb_g[k], w, C_g);
            C_b = mad(kb_b[k], w, C_b);
            T = test_T;
        }
    }

    if (px < width && py < height) {
        int pix = py * width + px;
        out_image[pix * 3 + 0] = C_r + T * bg_r;
        out_image[pix * 3 + 1] = C_g + T * bg_g;
        out_image[pix * 3 + 2] = C_b + T * bg_b;
    }
}
)CL";
