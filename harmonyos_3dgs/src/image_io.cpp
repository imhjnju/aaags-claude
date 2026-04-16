#include "image_io.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <string>

bool writePPM(const char* path, const float* image, int w, int h) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int i = 0; i < w * h; i++) {
        for (int c = 0; c < 3; c++) {
            float v = std::max(0.0f, std::min(1.0f, image[i * 3 + c]));
            uint8_t byte = (uint8_t)(v * 255.0f + 0.5f);
            fwrite(&byte, 1, 1, f);
        }
    }
    fclose(f);
    return true;
}

std::vector<float> readPPM(const char* path, int& width, int& height) {
    FILE* f = fopen(path, "rb");
    if (!f) return {};

    // Read magic
    char magic[3] = {};
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) {
        fclose(f);
        return {};
    }

    // Skip comments
    int ch;
    while ((ch = fgetc(f)) != EOF) {
        if (ch == '#') {
            while ((ch = fgetc(f)) != EOF && ch != '\n') {}
        } else if (ch > ' ') {
            ungetc(ch, f);
            break;
        }
    }

    int w, h, maxval;
    if (fscanf(f, "%d %d %d", &w, &h, &maxval) != 3) {
        fclose(f);
        return {};
    }
    // Consume the single whitespace after maxval
    fgetc(f);

    width = w;
    height = h;
    float scale = 1.0f / (float)maxval;

    std::vector<uint8_t> raw(w * h * 3);
    size_t read = fread(raw.data(), 1, raw.size(), f);
    fclose(f);
    if (read != raw.size()) return {};

    std::vector<float> image(w * h * 3);
    for (size_t i = 0; i < raw.size(); i++) {
        image[i] = (float)raw[i] * scale;
    }
    return image;
}
