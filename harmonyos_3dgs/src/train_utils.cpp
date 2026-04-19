#include "train_utils.h"
#include "train_types.h"  // lr_schedule

void build_L(const float rot[4], const float act_scale[3], float L[3][3]) {
    const float w = rot[0], x = rot[1], y = rot[2], z = rot[3];
    const float R[3][3] = {
        {1.f - 2.f*(y*y + z*z), 2.f*(x*y + w*z),       2.f*(x*z - w*y)},
        {2.f*(x*y - w*z),       1.f - 2.f*(x*x + z*z), 2.f*(y*z + w*x)},
        {2.f*(x*z + w*y),       2.f*(y*z - w*x),       1.f - 2.f*(x*x + y*y)}
    };
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            L[i][j] = R[i][j] * act_scale[j];
}

float spatial_lr_schedule(float lr_init, float lr_final,
                           float spatial_lr_scale,
                           int step, int max_steps) {
    return spatial_lr_scale * lr_schedule(lr_init, lr_final, step, max_steps);
}
