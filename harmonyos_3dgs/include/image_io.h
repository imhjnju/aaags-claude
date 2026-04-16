#pragma once
#include <vector>

bool writePPM(const char* path, const float* image, int width, int height);

// Read a PPM (P6 binary) image. Returns float [H*W*3] in [0,1].
// Sets width/height via output params. Returns empty vector on failure.
std::vector<float> readPPM(const char* path, int& width, int& height);
