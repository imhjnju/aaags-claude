#include "loss.h"
#include <cmath>

float l1_loss(const float* rendered, const float* gt, int H, int W, float* d_image) {
    int n = H * W * 3;
    float inv_n = 1.0f / (float)n;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = rendered[i] - gt[i];
        sum += std::abs(diff);
        if (diff > 0.0f)      d_image[i] =  inv_n;
        else if (diff < 0.0f) d_image[i] = -inv_n;
        else                  d_image[i] =  0.0f;
    }
    return sum * inv_n;
}

double l1_loss_double(const float* rendered, const float* gt, int H, int W, float* d_image) {
    int n = H * W * 3;
    float inv_n = 1.0f / (float)n;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        float diff = rendered[i] - gt[i];
        sum += std::abs((double)diff);
        if (diff > 0.0f)      d_image[i] =  inv_n;
        else if (diff < 0.0f) d_image[i] = -inv_n;
        else                  d_image[i] =  0.0f;
    }
    return sum / n;
}
