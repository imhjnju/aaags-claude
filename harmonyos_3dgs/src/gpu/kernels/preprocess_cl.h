// Auto-generated from preprocess.cl -- do not edit
static const char preprocess_cl_src[] = R"CL(
// preprocess.cl -- OpenCL kernel for 3DGS preprocessing (projection + cov + SH)
// Each work-item processes ONE Gaussian.
// Supports both standard 2D and AAA-Gaussians 3D evaluation paths.
// All matrices are column-major (matching CUDA/GLM convention).

// ---------------------------------------------------------------------------
// SH constants (matching sh_eval.cpp)
// ---------------------------------------------------------------------------
#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f

#define SH_C2_0  1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2  0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4  0.5462742152960396f

#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1  2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3  0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5  1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

#define ALPHA_THRESHOLD (1.0f / 255.0f)
#define Z_NEAR (-1.0f)
#define Z_FAR  (1.0f)

// ---------------------------------------------------------------------------
// Helper: sq
// ---------------------------------------------------------------------------
inline float sq(float x) { return x * x; }

// ---------------------------------------------------------------------------
// transformPoint4x3 -- column-major 4x4 matrix * point (ignore row 3)
// ---------------------------------------------------------------------------
inline void transformPoint4x3(const float* p, __constant const float* m, float out[3]) {
    out[0] = m[0] * p[0] + m[4] * p[1] + m[8]  * p[2] + m[12];
    out[1] = m[1] * p[0] + m[5] * p[1] + m[9]  * p[2] + m[13];
    out[2] = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
}

// ---------------------------------------------------------------------------
// transformPoint4x4 -- column-major 4x4 matrix * point (full)
// ---------------------------------------------------------------------------
inline void transformPoint4x4(const float* p, __constant const float* m, float out[4]) {
    out[0] = m[0] * p[0] + m[4] * p[1] + m[8]  * p[2] + m[12];
    out[1] = m[1] * p[0] + m[5] * p[1] + m[9]  * p[2] + m[13];
    out[2] = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
    out[3] = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
}

// ---------------------------------------------------------------------------
// ndc2Pix
// ---------------------------------------------------------------------------
inline float ndc2Pix(float v, int S) {
    return ((v + 1.0f) * (float)S - 1.0f) * 0.5f;
}

// ---------------------------------------------------------------------------
// mat4MulColMajor -- multiply two column-major 4x4 matrices
// ---------------------------------------------------------------------------
inline void mat4MulColMajor(__constant const float* A, const float* B, float out[16]) {
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++) {
            float sum = 0;
            for (int k = 0; k < 4; k++)
                sum += A[k*4 + row] * B[col*4 + k];
            out[col*4 + row] = sum;
        }
}
inline void mat4MulLocal(const float* A, const float* B, float out[16]) {
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++) {
            float sum = 0;
            for (int k = 0; k < 4; k++)
                sum += A[k*4 + row] * B[col*4 + k];
            out[col*4 + row] = sum;
        }
}

// ---------------------------------------------------------------------------
// quat2mat -- quaternion to rotation matrix (row-major)
// ---------------------------------------------------------------------------
inline void quat2mat(const float rot[4], float R[9]) {
    float r = rot[0], x = rot[1], y = rot[2], z = rot[3];
    R[0] = 1.f - 2.f*(y*y + z*z); R[1] = 2.f*(x*y + r*z);       R[2] = 2.f*(x*z - r*y);
    R[3] = 2.f*(x*y - r*z);       R[4] = 1.f - 2.f*(x*x + z*z); R[5] = 2.f*(y*z + r*x);
    R[6] = 2.f*(x*z + r*y);       R[7] = 2.f*(y*z - r*x);       R[8] = 1.f - 2.f*(x*x + y*y);
}

// ---------------------------------------------------------------------------
// inScreenRange
// ---------------------------------------------------------------------------
inline bool inScreenRange(const float pos_screen[4], const float range_from[3],
                          const float extent[3], int dim) {
    float d = pos_screen[dim] - range_from[dim] * pos_screen[3];
    return d > 0.0f && d < extent[dim] * pos_screen[3];
}
inline bool inScreenRangeD(const float pos_screen[4], const float range_from[3],
                           const float extent[3], int dim, float* d_out) {
    *d_out = pos_screen[dim] - range_from[dim] * pos_screen[3];
    return *d_out > 0.0f && *d_out < extent[dim] * pos_screen[3];
}

// ---------------------------------------------------------------------------
// maxContribRay
// ---------------------------------------------------------------------------
inline float maxContribRay(const float pa[4], const float pb[4], float mp[4]) {
    float dx = pa[1]*pb[2] - pa[2]*pb[1];
    float dy = pa[2]*pb[0] - pa[0]*pb[2];
    float dz = pa[0]*pb[1] - pa[1]*pb[0];
    float mx = pa[3]*pb[0] - pa[0]*pb[3];
    float my = pa[3]*pb[1] - pa[1]*pb[3];
    float mz = pa[3]*pb[2] - pa[2]*pb[3];
    float dd = dx*dx + dy*dy + dz*dz;
    float mdx = mx/dd, mdy = my/dd, mdz = mz/dd;
    mp[0] = dy*mdz - dz*mdy;
    mp[1] = dz*mdx - dx*mdz;
    mp[2] = dx*mdy - dy*mdx;
    mp[3] = 1.0f;
    return mx*mdx + my*mdy + mz*mdz;
}

// ---------------------------------------------------------------------------
// maxContribPlane
// ---------------------------------------------------------------------------
inline float maxContribPlane(const float plane[4], const float g2s[16], float mps[4]) {
    float norm = plane[3] / (plane[0]*plane[0] + plane[1]*plane[1] + plane[2]*plane[2]);
    float mpg[4] = {-plane[0]*norm, -plane[1]*norm, -plane[2]*norm, -plane[3]*norm};
    for (int i = 0; i < 4; i++) {
        mps[i] = 0;
        for (int j = 0; j < 4; j++)
            mps[i] += g2s[j*4+i] * mpg[j];
    }
    return -mpg[3];
}

// ---------------------------------------------------------------------------
// maxContribRayScreen
// ---------------------------------------------------------------------------
inline float maxContribRayScreen(const float pa[4], const float pb[4],
                                  const float g2s[16], float mps[4]) {
    float mpg[4];
    float contrib = maxContribRay(pa, pb, mpg);
    for (int i = 0; i < 4; i++) {
        mps[i] = 0;
        for (int j = 0; j < 4; j++)
            mps[i] += g2s[j*4+i] * mpg[j];
    }
    return contrib;
}

// ---------------------------------------------------------------------------
// maxContribGaussianFrustum3D
// ---------------------------------------------------------------------------
inline float maxContribGaussianFrustum3D(float smin_x, float smin_y, float smax_x, float smax_y,
                                          const float g2s[16], float* max_depth) {
    float mc = 1e30f;
    float mps[4] = {1,1,1,1};
    float RF[3] = {smin_x, smin_y, Z_NEAR};
    float EX[3] = {smax_x - smin_x, smax_y - smin_y, Z_FAR - Z_NEAR};
    float ms[4] = {g2s[3], g2s[7], g2s[11], g2s[15]};
    float d[3];
    bool bx = inScreenRangeD(ms, RF, EX, 0, &d[0]);
    bool by = inScreenRangeD(ms, RF, EX, 1, &d[1]);
    bool bz = inScreenRangeD(ms, RF, EX, 2, &d[2]);
    if (bx && by && bz) {
        for (int i=0;i<4;i++) mps[i]=ms[i];
        mc = 0.0f;
    } else {
        float dxs = (d[0] - EX[0]*0.5f*ms[3]) >= 0 ? EX[0]*0.5f : -EX[0]*0.5f;
        float dys = (d[1] - EX[1]*0.5f*ms[3]) >= 0 ? EX[1]*0.5f : -EX[1]*0.5f;
        float cx = RF[0]+EX[0]*0.5f+dxs, cy = RF[1]+EX[1]*0.5f+dys;
        float cpx[4], cpy[4];
        for (int i=0;i<4;i++) { cpx[i]=g2s[0*4+i]-g2s[3*4+i]*cx; cpy[i]=g2s[1*4+i]-g2s[3*4+i]*cy; }
        float ps[4]; float c;
        c = maxContribPlane(cpx, g2s, ps);
        if (c<mc && inScreenRange(ps,RF,EX,1) && inScreenRange(ps,RF,EX,2)) { mc=c; for(int i=0;i<4;i++) mps[i]=ps[i]; }
        c = maxContribPlane(cpy, g2s, ps);
        if (c<mc && inScreenRange(ps,RF,EX,0) && inScreenRange(ps,RF,EX,2)) { mc=c; for(int i=0;i<4;i++) mps[i]=ps[i]; }
        c = maxContribRayScreen(cpx, cpy, g2s, ps);
        if (c<mc && inScreenRange(ps,RF,EX,2)) { mc=c; for(int i=0;i<4;i++) mps[i]=ps[i]; }
        float opy[4]; float ocy=RF[1]+EX[1]*0.5f-dys;
        for(int i=0;i<4;i++) opy[i]=g2s[1*4+i]-g2s[3*4+i]*ocy;
        c = maxContribRayScreen(cpx, opy, g2s, ps);
        if (c<mc && inScreenRange(ps,RF,EX,2)) { mc=c; for(int i=0;i<4;i++) mps[i]=ps[i]; }
        float opx[4]; float ocx=RF[0]+EX[0]*0.5f-dxs;
        for(int i=0;i<4;i++) opx[i]=g2s[0*4+i]-g2s[3*4+i]*ocx;
        c = maxContribRayScreen(opx, cpy, g2s, ps);
        if (c<mc && inScreenRange(ps,RF,EX,2)) { mc=c; for(int i=0;i<4;i++) mps[i]=ps[i]; }
    }
    *max_depth = mps[2] / mps[3];
    return 0.5f * mc;
}

// ---------------------------------------------------------------------------
// normalizeAngle
// ---------------------------------------------------------------------------
inline float normalizeAngle(float theta) {
    theta = fmod(theta, 2.0f * M_PI_F);
    if (theta > M_PI_F) theta -= 2.0f * M_PI_F;
    if (theta <= -M_PI_F) theta += 2.0f * M_PI_F;
    return theta;
}

// ---------------------------------------------------------------------------
// computeAABBView
// ---------------------------------------------------------------------------
inline bool computeAABBView(const float g2v[16], const float mv[3],
                             float fx, float fy, float W, float H, float cutoff,
                             float mean2D[2], float extent2D[2]) {
    float t[4] = {cutoff, cutoff, cutoff, -1.0f};
    float vlen = sqrt(mv[0]*mv[0]+mv[1]*mv[1]+mv[2]*mv[2]);
    float vd[3] = {mv[0]/vlen, mv[1]/vlen, mv[2]/vlen};

    // For each axis (0=x, 1=y)
    for (int axis = 0; axis < 2; axis++) {
        float theta_mu = atan2(axis==0 ? vd[0] : vd[1], vd[2]);
        const float* Ta = g2v + axis*4;
        const float* T2 = g2v + 2*4;
        float sa=0, sz=0, mid_val=0;
        for (int i=0;i<4;i++) { sa += t[i]*Ta[i]*Ta[i]; sz += t[i]*T2[i]*T2[i]; mid_val += t[i]*Ta[i]*T2[i]; }
        float isq = mid_val*mid_val - sz*sa;
        float rlo = -(M_PI_2_F - 1e-5f), rhi = M_PI_2_F - 1e-5f;
        if (isq > 0.0f) {
            float sv = sqrt(isq);
            float ta0 = atan2(-(mid_val+sv), -sz);
            float ta1 = atan2(-(mid_val-sv), -sz);
            while (ta0 > theta_mu) ta0 -= M_PI_F;
            while (ta0 < (theta_mu - M_PI_F)) ta0 += M_PI_F;
            while (ta1 < theta_mu) ta1 += M_PI_F;
            while (ta1 > (theta_mu + M_PI_F)) ta1 -= M_PI_F;
            float n0 = normalizeAngle(ta0), n1 = normalizeAngle(ta1);
            if (theta_mu < 0.0f && fabs(n0) < fabs(n1)) { ta0 += 2.0f*M_PI_F; ta1 += 2.0f*M_PI_F; }
            else if (theta_mu > 0.0f && fabs(n1) < fabs(n0)) { ta0 -= 2.0f*M_PI_F; ta1 -= 2.0f*M_PI_F; }
            rlo = fmax(rlo, ta0); rhi = fmin(rhi, ta1);
        }
        float blo = (axis==0 ? W : H)/2.0f + (axis==0 ? fx : fy)*tan(rlo);
        float bhi = (axis==0 ? W : H)/2.0f + (axis==0 ? fx : fy)*tan(rhi);
        mean2D[axis] = (bhi+blo)/2.0f;
        extent2D[axis] = (bhi-blo)/2.0f;
    }
    return true;
}

// ---------------------------------------------------------------------------
// computeCov3D
// ---------------------------------------------------------------------------
inline void computeCov3D(const float scale[3], float mod, const float rot[4], float cov3D[6]) {
    float sx = mod * scale[0], sy = mod * scale[1], sz = mod * scale[2];
    float r = rot[0], x = rot[1], y = rot[2], z = rot[3];
    float R00 = 1.f - 2.f*(y*y+z*z), R01 = 2.f*(x*y+r*z),       R02 = 2.f*(x*z-r*y);
    float R10 = 2.f*(x*y-r*z),       R11 = 1.f - 2.f*(x*x+z*z), R12 = 2.f*(y*z+r*x);
    float R20 = 2.f*(x*z+r*y),       R21 = 2.f*(y*z-r*x),       R22 = 1.f - 2.f*(x*x+y*y);
    float M00=sx*R00, M01=sx*R01, M02=sx*R02;
    float M10=sy*R10, M11=sy*R11, M12=sy*R12;
    float M20=sz*R20, M21=sz*R21, M22=sz*R22;
    cov3D[0]=M00*M00+M10*M10+M20*M20; cov3D[1]=M00*M01+M10*M11+M20*M21;
    cov3D[2]=M00*M02+M10*M12+M20*M22; cov3D[3]=M01*M01+M11*M11+M21*M21;
    cov3D[4]=M01*M02+M11*M12+M21*M22; cov3D[5]=M02*M02+M12*M12+M22*M22;
}

// ---------------------------------------------------------------------------
// computeCov2D
// ---------------------------------------------------------------------------
inline void computeCov2D(const float mean3d[3], const float cov3D[6],
                         __constant const float* vm,
                         float focal_x, float focal_y,
                         float tan_fovx, float tan_fovy, float cov2D[3]) {
    float t[3];
    transformPoint4x3(mean3d, vm, t);
    float limx = 1.3f * tan_fovx, limy = 1.3f * tan_fovy;
    float txtz = t[0]/t[2], tytz = t[1]/t[2];
    t[0] = fmin(limx, fmax(-limx, txtz)) * t[2];
    t[1] = fmin(limy, fmax(-limy, tytz)) * t[2];
    float J00=focal_x/t[2], J02=-(focal_x*t[0])/(t[2]*t[2]);
    float J11=focal_y/t[2], J12=-(focal_y*t[1])/(t[2]*t[2]);
    float W00=vm[0],W01=vm[4],W02=vm[8]; float W10=vm[1],W11=vm[5],W12=vm[9]; float W20=vm[2],W21=vm[6],W22=vm[10];
    float T00=W00*J00+W20*J02, T01=W01*J00+W21*J02, T02=W02*J00+W22*J02;
    float T10=W10*J11+W20*J12, T11=W11*J11+W21*J12, T12=W12*J11+W22*J12;
    float Vrk00=cov3D[0],Vrk01=cov3D[1],Vrk02=cov3D[2];
    float Vrk10=cov3D[1],Vrk11=cov3D[3],Vrk12=cov3D[4];
    float Vrk20=cov3D[2],Vrk21=cov3D[4],Vrk22=cov3D[5];
    float tmp00=Vrk00*T00+Vrk01*T01+Vrk02*T02, tmp01=Vrk10*T00+Vrk11*T01+Vrk12*T02, tmp02=Vrk20*T00+Vrk21*T01+Vrk22*T02;
    float tmp10=Vrk00*T10+Vrk01*T11+Vrk02*T12, tmp11=Vrk10*T10+Vrk11*T11+Vrk12*T12, tmp12=Vrk20*T10+Vrk21*T11+Vrk22*T12;
    cov2D[0]=T00*tmp00+T01*tmp01+T02*tmp02;
    cov2D[1]=T00*tmp10+T01*tmp11+T02*tmp12;
    cov2D[2]=T10*tmp10+T11*tmp11+T12*tmp12;
}

// ---------------------------------------------------------------------------
// computeColorFromSH
// ---------------------------------------------------------------------------
inline void computeColorFromSH(int degree, int max_coeffs,
                                __global const float* sh_coeffs,
                                const float pos[3], __constant const float* cam_pos,
                                float rgb_out[3]) {
    float dx = pos[0]-cam_pos[0], dy = pos[1]-cam_pos[1], dz = pos[2]-cam_pos[2];
    float len = sqrt(dx*dx+dy*dy+dz*dz);
    dx/=len; dy/=len; dz/=len;
    float x=dx, y=dy, z=dz;
    rgb_out[0]=SH_C0*sh_coeffs[0]; rgb_out[1]=SH_C0*sh_coeffs[1]; rgb_out[2]=SH_C0*sh_coeffs[2];
    if (degree > 0) {
        for (int ch=0;ch<3;ch++)
            rgb_out[ch] += -SH_C1*y*sh_coeffs[1*3+ch] + SH_C1*z*sh_coeffs[2*3+ch] + -SH_C1*x*sh_coeffs[3*3+ch];
        if (degree > 1) {
            float xx=x*x,yy=y*y,zz=z*z,xy=x*y,yz=y*z,xz=x*z;
            for (int ch=0;ch<3;ch++)
                rgb_out[ch] += SH_C2_0*xy*sh_coeffs[4*3+ch]+SH_C2_1*yz*sh_coeffs[5*3+ch]
                    +SH_C2_2*(2.0f*zz-xx-yy)*sh_coeffs[6*3+ch]+SH_C2_3*xz*sh_coeffs[7*3+ch]
                    +SH_C2_4*(xx-yy)*sh_coeffs[8*3+ch];
            if (degree > 2) {
                for (int ch=0;ch<3;ch++)
                    rgb_out[ch] += SH_C3_0*y*(3.0f*xx-yy)*sh_coeffs[9*3+ch]
                        +SH_C3_1*xy*z*sh_coeffs[10*3+ch]+SH_C3_2*y*(4.0f*zz-xx-yy)*sh_coeffs[11*3+ch]
                        +SH_C3_3*z*(2.0f*zz-3.0f*xx-3.0f*yy)*sh_coeffs[12*3+ch]
                        +SH_C3_4*x*(4.0f*zz-xx-yy)*sh_coeffs[13*3+ch]+SH_C3_5*z*(xx-yy)*sh_coeffs[14*3+ch]
                        +SH_C3_6*x*(xx-3.0f*yy)*sh_coeffs[15*3+ch];
            }
        }
    }
    rgb_out[0]=fmin(1.0f,fmax(0.0f,rgb_out[0]+0.5f));
    rgb_out[1]=fmin(1.0f,fmax(0.0f,rgb_out[1]+0.5f));
    rgb_out[2]=fmin(1.0f,fmax(0.0f,rgb_out[2]+0.5f));
}

// ---------------------------------------------------------------------------
// Main preprocess kernel -- one work-item per Gaussian
// ---------------------------------------------------------------------------
__kernel void preprocess(
    __global const float* positions,
    __global const float* sh_coeffs,
    __global const float* scales,
    __global const float* rotations,
    __global const float* opacities,
    __constant const float* view_matrix,
    __constant const float* viewproj,
    __constant const float* cam_pos,
    const int N,
    const int width,
    const int height,
    const float tan_fovx,
    const float tan_fovy,
    const float scale_modifier,
    const int sh_degree,
    const int max_coeffs,
    const int tile_w,
    const int tile_h,
    const int antialiasing,
    __global float* out_means2D,
    __global float* out_depths,
    __global float* out_conics,
    __global float* out_rgb,
    __global float* out_opacities_2d,
    __global int*   out_radii,
    __global int*   out_tiles_touched,
    // AAA-Gaussians: eval_3D mode
    const int eval_3D,
    __global const float* filter_3D,     // [N] or NULL
    __global float* out_gauss2screen,    // [N*16] or NULL
    __global float* out_cov3D_inv,       // [N*6] eval_3D: inverse 3D covariance
    __global float* out_mean_offset      // [N*3] eval_3D: pos - cam_pos
)
{
    int i = get_global_id(0);
    if (i >= N) return;

    out_radii[i] = 0;
    out_tiles_touched[i] = 0;

    float pos[3];
    pos[0] = positions[i*3]; pos[1] = positions[i*3+1]; pos[2] = positions[i*3+2];

    float p_view[3];
    transformPoint4x3(pos, view_matrix, p_view);
    if (p_view[2] <= 0.2f) return;

    float focal_x = (float)width / (2.0f * tan_fovx);
    float focal_y = (float)height / (2.0f * tan_fovy);
    int grid_x = (width + tile_w - 1) / tile_w;
    int grid_y = (height + tile_h - 1) / tile_h;

    if (eval_3D) {
        // === AAA-Gaussians 3D evaluation path ===
        float opacity = opacities[i];
        float focal = fmax(focal_x, focal_y);
        float f3d = (filter_3D != 0) ? filter_3D[i] : 0.0f;

        float rot[4] = {rotations[i*4], rotations[i*4+1], rotations[i*4+2], rotations[i*4+3]};
        float scl[3] = {scales[i*3], scales[i*3+1], scales[i*3+2]};

        // Quaternion to rotation matrix
        float R[9];
        quat2mat(rot, R);

        // Compute dilated scale
        float scale_dilated[3] = {scl[0], scl[1], scl[2]};
        float dilation_factor = 1.0f;

        float kernel_size = 0.3f;
        {
            float vr[3] = {pos[0]-cam_pos[0], pos[1]-cam_pos[1], pos[2]-cam_pos[2]};
            float vrl = sqrt(vr[0]*vr[0]+vr[1]*vr[1]+vr[2]*vr[2]);
            vr[0]/=vrl; vr[1]/=vrl; vr[2]/=vrl;

            float vrg[3];
            for (int j=0;j<3;j++) vrg[j] = R[j*3]*vr[0]+R[j*3+1]*vr[1]+R[j*3+2]*vr[2];
            float rsq[3] = {vrg[0]*vrg[0], vrg[1]*vrg[1], vrg[2]*vrg[2]};

            float smp = sq(p_view[2] / focal) * kernel_size;
            smp = fmax(sq(f3d), smp);

            for (int j=0;j<3;j++) scale_dilated[j] = sqrt(scl[j]*scl[j] + smp);

            float s[3] = {scl[0]*scl[0], scl[1]*scl[1], scl[2]*scl[2]};
            float sd[3] = {scale_dilated[0]*scale_dilated[0], scale_dilated[1]*scale_dilated[1], scale_dilated[2]*scale_dilated[2]};
            float dm = rsq[0]*(s[1]*s[2])+rsq[1]*(s[2]*s[0])+rsq[2]*(s[0]*s[1]);
            float dmd = rsq[0]*(sd[1]*sd[2])+rsq[1]*(sd[2]*sd[0])+rsq[2]*(sd[0]*sd[1]);
            dilation_factor = sqrt(dm/dmd);
        }

        opacity *= dilation_factor;
        if (opacity < ALPHA_THRESHOLD) return;

        // Build L = transpose(S * R)
        float sm = sqrt(scale_modifier);
        float sd[3] = {scale_dilated[0]*sm, scale_dilated[1]*sm, scale_dilated[2]*sm};
        // Compute inverse 3D covariance: R * diag(1/sd^2) * R^T
        {
            float inv_sd2[3];
            for (int j=0;j<3;j++) inv_sd2[j] = 1.0f/(sd[j]*sd[j]);
            int ci_idx = 0;
            for (int r=0;r<3;r++) {
                for (int c=r;c<3;c++) {
                    float v = 0;
                    for (int k=0;k<3;k++) v += R[k*3+r]*inv_sd2[k]*R[k*3+c];
                    out_cov3D_inv[i*6+ci_idx] = v;
                    ci_idx++;
                }
            }
        }
        // Store mean_offset = pos - cam_pos
        out_mean_offset[i*3]   = pos[0] - cam_pos[0];
        out_mean_offset[i*3+1] = pos[1] - cam_pos[1];
        out_mean_offset[i*3+2] = pos[2] - cam_pos[2];

        float L[9]; // L[row][col] = sd[col] * R[col*3+row]
        for (int r=0;r<3;r++) for (int c=0;c<3;c++) L[r*3+c] = sd[c] * R[c*3+r];

        // gauss2world (column-major)
        float g2w[16];
        for (int col=0;col<3;col++) { g2w[col*4]=L[col]; g2w[col*4+1]=L[3+col]; g2w[col*4+2]=L[6+col]; g2w[col*4+3]=0; }
        g2w[12]=pos[0]; g2w[13]=pos[1]; g2w[14]=pos[2]; g2w[15]=1.0f;

        // viewport (column-major)
        float vp[16] = {0};
        vp[0]=(float)width/2.0f; vp[5]=(float)height/2.0f; vp[10]=1.0f; vp[15]=1.0f;
        vp[12]=(float)width/2.0f-0.5f; vp[13]=(float)height/2.0f-0.5f;

        // world2screen = vp * viewproj
        float w2s[16];
        mat4MulColMajor(viewproj, g2w, w2s); // temp: viewproj*g2w
        // Actually: w2s = vp * viewproj first
        float vp_vp[16];
        // We need: gauss2screen = viewport * viewproj * gauss2world
        // Step 1: tmp = viewproj * gauss2world
        float tmp[16];
        mat4MulColMajor(viewproj, g2w, tmp);
        // Step 2: g2s = viewport * tmp
        float g2s_cm[16];
        mat4MulLocal(vp, tmp, g2s_cm);

        // gauss2view = view_matrix * gauss2world
        float g2v_cm[16];
        mat4MulColMajor(view_matrix, g2w, g2v_cm);

        // Store as row-major
        float g2s[16], g2v[16];
        for (int r=0;r<4;r++) for (int c=0;c<4;c++) { g2s[r*4+c]=g2s_cm[c*4+r]; g2v[r*4+c]=g2v_cm[c*4+r]; }

        // Compute campos_gauss for inside-ellipsoid check
        float diff[3] = {cam_pos[0]-pos[0], cam_pos[1]-pos[1], cam_pos[2]-pos[2]};
        float cg[3];
        for (int j=0;j<3;j++) cg[j] = (R[j*3]*diff[0]+R[j*3+1]*diff[1]+R[j*3+2]*diff[2]) / sd[j];
        float dist_sq = cg[0]*cg[0]+cg[1]*cg[1]+cg[2]*cg[2];

        float opt = log(opacity / ALPHA_THRESHOLD);
        float cutoff = fmin(11.11f, 2.0f * opt);
        if (dist_sq < cutoff) return;

        // 3D frustum culling
        float max_depth;
        float mc = maxContribGaussianFrustum3D(0, 0, (float)(width-1), (float)(height-1), g2s, &max_depth);
        if (mc > opt) return;

        // NDC-projected center for tile assignment (consistent with 2D path)
        float ph[4];
        transformPoint4x4(pos, viewproj, ph);
        float pw_inv = 1.0f / (ph[3] + 1e-7f);
        float pixel_x = ndc2Pix(ph[0]*pw_inv, width);
        float pixel_y = ndc2Pix(ph[1]*pw_inv, height);

        // Screen-space AABB for extent (with view-space fallback)
        float e2d[2];
        {
            float t_v[4] = {cutoff, cutoff, cutoff, -1.0f};
            const float* ar0 = g2s;
            const float* ar1 = g2s + 4;
            const float* ar3 = g2s + 12;
            float ss = 0;
            for (int k=0;k<4;k++) ss += t_v[k]*ar3[k]*ar3[k];
            if (ss < 0.0f) {
                // Screen-space AABB succeeded
                float ff[4]; for (int k=0;k<4;k++) ff[k] = t_v[k]/ss;
                float pp[2]={0,0};
                for (int k=0;k<4;k++) { pp[0]+=ff[k]*ar0[k]*ar3[k]; pp[1]+=ff[k]*ar1[k]*ar3[k]; }
                float hh[2]={pp[0]*pp[0], pp[1]*pp[1]};
                for (int k=0;k<4;k++) { hh[0]-=ff[k]*ar0[k]*ar0[k]; hh[1]-=ff[k]*ar1[k]*ar1[k]; }
                e2d[0]=sqrt(fmax(0.0f,hh[0])); e2d[1]=sqrt(fmax(0.0f,hh[1]));
            } else {
                // Fallback: view-space AABB
                // g2v is the gauss2view matrix (row-major), computed earlier
                bool view_ok = true;
                for (int axis = 0; axis < 2; axis++) {
                    const float* Ta = g2v + axis * 4;
                    const float* T2 = g2v + 2 * 4;
                    float sa = 0, sz = 0, mid_v = 0;
                    for (int k = 0; k < 4; k++) {
                        sa += t_v[k] * Ta[k] * Ta[k];
                        sz += t_v[k] * T2[k] * T2[k];
                        mid_v += t_v[k] * Ta[k] * T2[k];
                    }
                    float disc = mid_v * mid_v - sz * sa;
                    if (disc <= 0.0f) {
                        e2d[axis] = (axis == 0) ? (float)width * 0.5f : (float)height * 0.5f;
                    } else {
                        float sq_d = sqrt(disc);
                        float th0 = atan2(-(mid_v + sq_d), -sz);
                        float th1 = atan2(-(mid_v - sq_d), -sz);
                        float foc = (axis == 0) ? focal_x : focal_y;
                        float dim = (axis == 0) ? (float)width : (float)height;
                        float b0 = dim * 0.5f + foc * tan(th0);
                        float b1 = dim * 0.5f + foc * tan(th1);
                        if (b0 > b1) { float tmp = b0; b0 = b1; b1 = tmp; }
                        e2d[axis] = (b1 - b0) * 0.5f;
                    }
                }
                if (!view_ok) return;
            }
        }

        int my_radius = max(1, (int)ceil(fmax(e2d[0], e2d[1])));
        int rmx = min(grid_x, max(0, (int)((pixel_x-(float)my_radius)/(float)tile_w)));
        int rmy = min(grid_y, max(0, (int)((pixel_y-(float)my_radius)/(float)tile_h)));
        int rMx = min(grid_x, max(0, (int)((pixel_x+(float)my_radius+(float)tile_w-1.0f)/(float)tile_w)));
        int rMy = min(grid_y, max(0, (int)((pixel_y+(float)my_radius+(float)tile_h-1.0f)/(float)tile_h)));
        if ((rMx-rmx)*(rMy-rmy)==0) return;

        // SH
        float rgb[3];
        computeColorFromSH(sh_degree, max_coeffs, &sh_coeffs[i*max_coeffs*3], pos, cam_pos, rgb);

        // Store — use view-space z for depth sorting
        out_depths[i] = p_view[2];
        out_radii[i] = my_radius;
        out_means2D[i*2] = pixel_x; out_means2D[i*2+1] = pixel_y;
        out_opacities_2d[i] = opacity;
        out_tiles_touched[i] = (rMy-rmy)*(rMx-rmx);
        out_rgb[i*3]=rgb[0]; out_rgb[i*3+1]=rgb[1]; out_rgb[i*3+2]=rgb[2];
        for (int j=0;j<16;j++) out_gauss2screen[i*16+j] = g2s[j];

    } else {
        // === Standard 2D 3DGS path ===
        float p_hom[4];
        transformPoint4x4(pos, viewproj, p_hom);
        float p_w = 1.0f / (p_hom[3] + 1e-7f);
        float p_ndc0 = p_hom[0]*p_w, p_ndc1 = p_hom[1]*p_w;
        float pixel_x = ndc2Pix(p_ndc0, width), pixel_y = ndc2Pix(p_ndc1, height);

        float scl[3] = {scales[i*3], scales[i*3+1], scales[i*3+2]};
        float rot[4] = {rotations[i*4], rotations[i*4+1], rotations[i*4+2], rotations[i*4+3]};
        float cov3d[6]; computeCov3D(scl, scale_modifier, rot, cov3d);
        float cov2d[3]; computeCov2D(pos, cov3d, view_matrix, focal_x, focal_y, tan_fovx, tan_fovy, cov2d);

        float det_cov = cov2d[0]*cov2d[2]-cov2d[1]*cov2d[1];
        cov2d[0]+=0.3f; cov2d[2]+=0.3f;
        float det_cp = cov2d[0]*cov2d[2]-cov2d[1]*cov2d[1];
        float hcs = 1.0f;
        if (antialiasing) hcs = sqrt(fmax(0.000025f, det_cov/det_cp));
        if (det_cp == 0.0f) return;
        float di = 1.0f/det_cp;
        float c0=cov2d[2]*di, c1=-cov2d[1]*di, c2=cov2d[0]*di;
        float mid=0.5f*(cov2d[0]+cov2d[2]);
        float ml=fmax(mid+sqrt(fmax(0.1f,mid*mid-det_cp)), mid-sqrt(fmax(0.1f,mid*mid-det_cp)));
        int my_radius = (int)ceil(3.0f*sqrt(ml));
        if (my_radius > max(width,height)) return;
        int rmx=min(grid_x,max(0,(int)((pixel_x-(float)my_radius)/(float)tile_w)));
        int rmy=min(grid_y,max(0,(int)((pixel_y-(float)my_radius)/(float)tile_h)));
        int rMx=min(grid_x,max(0,(int)((pixel_x+(float)my_radius+(float)tile_w-1.0f)/(float)tile_w)));
        int rMy=min(grid_y,max(0,(int)((pixel_y+(float)my_radius+(float)tile_h-1.0f)/(float)tile_h)));
        if ((rMx-rmx)*(rMy-rmy)==0) return;
        float rgb[3]; computeColorFromSH(sh_degree, max_coeffs, &sh_coeffs[i*max_coeffs*3], pos, cam_pos, rgb);
        out_depths[i]=p_view[2]; out_radii[i]=my_radius;
        out_means2D[i*2]=pixel_x; out_means2D[i*2+1]=pixel_y;
        out_conics[i*3]=c0; out_conics[i*3+1]=c1; out_conics[i*3+2]=c2;
        out_rgb[i*3]=rgb[0]; out_rgb[i*3+1]=rgb[1]; out_rgb[i*3+2]=rgb[2];
        out_opacities_2d[i] = opacities[i]*hcs;
        out_tiles_touched[i] = (rMy-rmy)*(rMx-rmx);
    }
}
)CL";
