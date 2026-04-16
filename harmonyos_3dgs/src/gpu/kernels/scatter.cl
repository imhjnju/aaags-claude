// scatter.cl -- Generate (tile_id << 32 | depth_bits) keys for each Gaussian-tile pair
// One work-item per Gaussian. Supports per-tile 3D frustum culling for eval_3D mode.

#define Z_NEAR_SC (-1.0f)
#define Z_FAR_SC  (1.0f)

// --- Per-tile 3D frustum culling helpers (inlined from preprocess.cl) ---

inline bool inScreenRangeSC(const float ps[4], const float rf[3], const float ex[3], int dim) {
    float d = ps[dim] - rf[dim] * ps[3];
    return d > 0.0f && d < ex[dim] * ps[3];
}
inline bool inScreenRangeDSC(const float ps[4], const float rf[3], const float ex[3], int dim, float* d_out) {
    *d_out = ps[dim] - rf[dim] * ps[3];
    return *d_out > 0.0f && *d_out < ex[dim] * ps[3];
}

inline float maxContribRaySC(const float pa[4], const float pb[4], float mp[4]) {
    float dx=pa[1]*pb[2]-pa[2]*pb[1], dy=pa[2]*pb[0]-pa[0]*pb[2], dz=pa[0]*pb[1]-pa[1]*pb[0];
    float mx=pa[3]*pb[0]-pa[0]*pb[3], my=pa[3]*pb[1]-pa[1]*pb[3], mz=pa[3]*pb[2]-pa[2]*pb[3];
    float dd=dx*dx+dy*dy+dz*dz;
    float mdx=mx/dd, mdy=my/dd, mdz=mz/dd;
    mp[0]=dy*mdz-dz*mdy; mp[1]=dz*mdx-dx*mdz; mp[2]=dx*mdy-dy*mdx; mp[3]=1.0f;
    return mx*mdx+my*mdy+mz*mdz;
}

inline float maxContribPlaneSC(const float pl[4], const float g2s[16], float mps[4]) {
    float nm = pl[3]/(pl[0]*pl[0]+pl[1]*pl[1]+pl[2]*pl[2]);
    float mpg[4] = {-pl[0]*nm, -pl[1]*nm, -pl[2]*nm, -pl[3]*nm};
    for (int i=0;i<4;i++) { mps[i]=0; for (int j=0;j<4;j++) mps[i]+=g2s[j*4+i]*mpg[j]; }
    return -mpg[3];
}

inline float maxContribRayScreenSC(const float pa[4], const float pb[4], const float g2s[16], float mps[4]) {
    float mpg[4]; float c = maxContribRaySC(pa, pb, mpg);
    for (int i=0;i<4;i++) { mps[i]=0; for (int j=0;j<4;j++) mps[i]+=g2s[j*4+i]*mpg[j]; }
    return c;
}

inline float maxContribFrustum3DSC(float smin_x, float smin_y, float smax_x, float smax_y,
                                    __global const float* g2s_ptr) {
    float g2s[16]; for (int i=0;i<16;i++) g2s[i]=g2s_ptr[i];
    float mc=1e30f; float mps[4]={1,1,1,1};
    float RF[3]={smin_x, smin_y, Z_NEAR_SC};
    float EX[3]={smax_x-smin_x, smax_y-smin_y, Z_FAR_SC-Z_NEAR_SC};
    float ms[4]={g2s[3],g2s[7],g2s[11],g2s[15]};
    float d[3];
    bool bx=inScreenRangeDSC(ms,RF,EX,0,&d[0]);
    bool by=inScreenRangeDSC(ms,RF,EX,1,&d[1]);
    bool bz=inScreenRangeDSC(ms,RF,EX,2,&d[2]);
    if (bx&&by&&bz) { mc=0.0f; }
    else {
        float dxs=(d[0]-EX[0]*0.5f*ms[3])>=0?EX[0]*0.5f:-EX[0]*0.5f;
        float dys=(d[1]-EX[1]*0.5f*ms[3])>=0?EX[1]*0.5f:-EX[1]*0.5f;
        float cx=RF[0]+EX[0]*0.5f+dxs, cy=RF[1]+EX[1]*0.5f+dys;
        float cpx[4],cpy[4]; for(int i=0;i<4;i++){cpx[i]=g2s[i]-g2s[12+i]*cx; cpy[i]=g2s[4+i]-g2s[12+i]*cy;}
        float ps[4]; float c;
        c=maxContribPlaneSC(cpx,g2s,ps); if(c<mc&&inScreenRangeSC(ps,RF,EX,1)&&inScreenRangeSC(ps,RF,EX,2)){mc=c;}
        c=maxContribPlaneSC(cpy,g2s,ps); if(c<mc&&inScreenRangeSC(ps,RF,EX,0)&&inScreenRangeSC(ps,RF,EX,2)){mc=c;}
        c=maxContribRayScreenSC(cpx,cpy,g2s,ps); if(c<mc&&inScreenRangeSC(ps,RF,EX,2)){mc=c;}
        float opy[4]; float ocy=RF[1]+EX[1]*0.5f-dys;
        for(int i=0;i<4;i++) opy[i]=g2s[4+i]-g2s[12+i]*ocy;
        c=maxContribRayScreenSC(cpx,opy,g2s,ps); if(c<mc&&inScreenRangeSC(ps,RF,EX,2)){mc=c;}
        float opx[4]; float ocx=RF[0]+EX[0]*0.5f-dxs;
        for(int i=0;i<4;i++) opx[i]=g2s[i]-g2s[12+i]*ocx;
        c=maxContribRayScreenSC(opx,cpy,g2s,ps); if(c<mc&&inScreenRangeSC(ps,RF,EX,2)){mc=c;}
    }
    return 0.5f*mc;
}

// --- Main scatter kernel ---

__kernel void scatter(
    __global const float* means2D,       // [N * 2]
    __global const int*   radii,         // [N]
    __global const float* depths,        // [N]
    __global const int*   point_offsets, // [N] exclusive prefix sum of tiles_touched
    const int N,
    const int grid_x,
    const int grid_y,
    const int tile_w,
    const int tile_h,
    __global ulong*    keys_out,         // [total_pairs]
    __global uint*     values_out,       // [total_pairs]
    // AAA-Gaussians per-tile culling
    const int eval_3D,
    __global const float* gauss2screen,  // [N*16] or NULL
    __global const float* opacities_2d,  // [N] or NULL
    // Per-tile depth key
    __global const float* cov3D_inv,     // [N*6] or NULL
    __global const float* mean_offset,   // [N*3] or NULL
    __constant const float* inverse_vp,  // [16] inverse viewproj matrix
    __constant const float* cam_pos_u    // [3] camera position
)
{
    int idx = get_global_id(0);
    if (idx >= N) return;
    if (radii[idx] <= 0) return;

    int off = point_offsets[idx];

    float px = means2D[idx * 2];
    float py = means2D[idx * 2 + 1];
    int my_radius = radii[idx];

    int rect_min_x = min(grid_x, max(0, (int)((px - (float)my_radius) / (float)tile_w)));
    int rect_min_y = min(grid_y, max(0, (int)((py - (float)my_radius) / (float)tile_h)));
    int rect_max_x = min(grid_x, max(0, (int)((px + (float)my_radius + (float)tile_w - 1.0f) / (float)tile_w)));
    int rect_max_y = min(grid_y, max(0, (int)((py + (float)my_radius + (float)tile_h - 1.0f) / (float)tile_h)));

    float d = depths[idx];
    uint depth_bits = as_uint(d);

    for (int y = rect_min_y; y < rect_max_y; y++) {
        for (int x = rect_min_x; x < rect_max_x; x++) {
            // Per-tile depth key via depthAlongRay (StopThePop corrects at pixel level)
            uint tile_depth_bits = depth_bits;
            if (eval_3D && cov3D_inv != 0) {
                float tcx = ((float)x + 0.5f) * (float)tile_w;
                float tcy = ((float)y + 0.5f) * (float)tile_h;
                float screen_w = (float)(grid_x * tile_w);
                float screen_h = (float)(grid_y * tile_h);
                float ndc_x = (2.0f * tcx + 1.0f) / screen_w - 1.0f;
                float ndc_y = (2.0f * tcy + 1.0f) / screen_h - 1.0f;
                float p4[4] = {ndc_x, ndc_y, 0.0f, 1.0f};
                float wp[4];
                for (int ii=0;ii<4;ii++)
                    wp[ii] = inverse_vp[ii]*p4[0]+inverse_vp[4+ii]*p4[1]+inverse_vp[8+ii]*p4[2]+inverse_vp[12+ii]*p4[3];
                float wi = 1.0f/(wp[3]+1e-10f);
                float vd[3] = {wp[0]*wi-cam_pos_u[0], wp[1]*wi-cam_pos_u[1], wp[2]*wi-cam_pos_u[2]};
                float vl = sqrt(vd[0]*vd[0]+vd[1]*vd[1]+vd[2]*vd[2]);
                if (vl > 1e-10f) { vd[0]/=vl; vd[1]/=vl; vd[2]/=vl; }
                float ci[6]; for (int j=0;j<6;j++) ci[j]=cov3D_inv[idx*6+j];
                float mo[3] = {mean_offset[idx*3], mean_offset[idx*3+1], mean_offset[idx*3+2]};
                float Sv[3];
                Sv[0]=ci[0]*vd[0]+ci[1]*vd[1]+ci[2]*vd[2];
                Sv[1]=ci[1]*vd[0]+ci[3]*vd[1]+ci[4]*vd[2];
                Sv[2]=ci[2]*vd[0]+ci[4]*vd[1]+ci[5]*vd[2];
                float num=mo[0]*Sv[0]+mo[1]*Sv[1]+mo[2]*Sv[2];
                float den=vd[0]*Sv[0]+vd[1]*Sv[1]+vd[2]*Sv[2];
                if (fabs(den) > 1e-10f) {
                    float ptd = num/den;
                    if (ptd > 0.0f) tile_depth_bits = as_uint(ptd);
                }
            }

            uint tile_id = (uint)(y * grid_x + x);
            ulong key = ((ulong)tile_id << 32) | (ulong)tile_depth_bits;
            keys_out[off] = key;
            values_out[off] = (uint)idx;
            off++;
        }
    }
}
