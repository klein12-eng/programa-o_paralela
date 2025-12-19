// cpuProcessor.cpp

#include "cpuProcessor.h"
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <vector>

cpuProcessor::cpuProcessor() {
}

cpuProcessor::~cpuProcessor() {
}

// Simple box blur on CPU (naive implementation)
void cpuProcessor::cpuapplyBlur(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius) {
    if (!h_in || !h_out || width <= 0 || height <= 0 || channels < 3) return;
    int kernelSize = 2 * radius + 1;
    int kernelArea = kernelSize * kernelSize;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int rAcc = 0, gAcc = 0, bAcc = 0, aAcc = 0, count = 0;
            for (int ky = -radius; ky <= radius; ++ky) {
                int ny = y + ky;
                if (ny < 0 || ny >= height) continue;
                for (int kx = -radius; kx <= radius; ++kx) {
                    int nx = x + kx;
                    if (nx < 0 || nx >= width) continue;
                    int idx = (ny * width + nx) * channels;
                    bAcc += h_in[idx + 0];
                    gAcc += h_in[idx + 1];
                    rAcc += h_in[idx + 2];
                    if (channels == 4) aAcc += h_in[idx + 3];
                    ++count;
                }
            }
            int outIdx = (y * width + x) * channels;
            h_out[outIdx + 0] = static_cast<unsigned char>(bAcc / count);
            h_out[outIdx + 1] = static_cast<unsigned char>(gAcc / count);
            h_out[outIdx + 2] = static_cast<unsigned char>(rAcc / count);
            if (channels == 4) h_out[outIdx + 3] = static_cast<unsigned char>(aAcc / count);
        }
    }
}

// Variants simply dispatch to the base blur for now
void cpuProcessor::cpuapplyBlur_a(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius) {
    cpuapplyBlur(h_in, h_out, width, height, channels, radius);
}

void cpuProcessor::cpuapplyBlur_b(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius) {
    cpuapplyBlur(h_in, h_out, width, height, channels, radius);
}

void cpuProcessor::cpuapplyBW(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels) {
    if (!h_in || !h_out || width <= 0 || height <= 0 || channels < 3) return;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            unsigned char b = h_in[idx + 0];
            unsigned char g = h_in[idx + 1];
            unsigned char r = h_in[idx + 2];
            int gray = static_cast<int>(0.299 * r + 0.587 * g + 0.114 * b);
            unsigned char gVal = static_cast<unsigned char>(std::min(255, gray));
            h_out[idx + 0] = gVal;
            h_out[idx + 1] = gVal;
            h_out[idx + 2] = gVal;
            if (channels == 4) {
                h_out[idx + 3] = h_in[idx + 3];
            }
        }
    }
}

// Build Gaussian kernel (1D) for separable convolution
static std::vector<double> buildGaussianKernel(int radius, double sigma) {
    int size = 2 * radius + 1;
    std::vector<double> k(size);
    double sum = 0.0;
    double s2 = 2.0 * sigma * sigma;
    for (int i = -radius; i <= radius; ++i) {
        double v = std::exp(-(i * i) / s2);
        k[i + radius] = v;
        sum += v;
    }
    for (int i = 0; i < size; ++i) k[i] /= sum;
    return k;
}

void cpuProcessor::cpuapplyGaussian(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius, double sigma) {
    if (!h_in || !h_out) return;
    if (radius <= 0) { // fallback to copy
        std::memcpy(h_out, h_in, (size_t)width * height * channels);
        return;
    }
    auto k = buildGaussianKernel(radius, sigma);
    int size = 2 * radius + 1;
    // temporary buffer for horizontal pass
    std::vector<unsigned char> tmp((size_t)width * height * channels);
    // horizontal pass
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                double acc = 0.0;
                for (int i = -radius; i <= radius; ++i) {
                    int nx = x + i;
                    if (nx < 0) nx = 0; if (nx >= width) nx = width - 1;
                    int idx = (y * width + nx) * channels + c;
                    acc += k[i + radius] * h_in[idx];
                }
                tmp[(y * width + x) * channels + c] = static_cast<unsigned char>(std::min(255.0, std::max(0.0, acc)));
            }
        }
    }
    // vertical pass
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                double acc = 0.0;
                for (int j = -radius; j <= radius; ++j) {
                    int ny = y + j;
                    if (ny < 0) ny = 0; if (ny >= height) ny = height - 1;
                    int idx = (ny * width + x) * channels + c;
                    acc += k[j + radius] * tmp[idx];
                }
                h_out[(y * width + x) * channels + c] = static_cast<unsigned char>(std::min(255.0, std::max(0.0, acc)));
            }
        }
    }
}

void cpuProcessor::cpuapplyEdge(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels) {
    if (!h_in || !h_out) return;
    // Use simple Sobel on luminance
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double gx = 0.0, gy = 0.0;
            for (int j = -1; j <= 1; ++j) {
                int ny = std::min(std::max(y + j, 0), height - 1);
                for (int i = -1; i <= 1; ++i) {
                    int nx = std::min(std::max(x + i, 0), width - 1);
                    int idx = (ny * width + nx) * channels;
                    int b = h_in[idx + 0];
                    int g = h_in[idx + 1];
                    int r = h_in[idx + 2];
                    double lum = 0.299 * r + 0.587 * g + 0.114 * b;
                    // Sobel kernels
                    int kx = 0;
                    if (j == -1) { if (i == -1) kx = -1; if (i == 0) kx = 0; if (i == 1) kx = 1; }
                    if (j == 0)  { if (i == -1) kx = -2; if (i == 0) kx = 0; if (i == 1) kx = 2; }
                    if (j == 1)  { if (i == -1) kx = -1; if (i == 0) kx = 0; if (i == 1) kx = 1; }
                    int ky = 0;
                    if (j == -1) { if (i == -1) ky = -1; if (i == 0) ky = -2; if (i == 1) ky = -1; }
                    if (j == 0)  { if (i == -1) ky = 0; if (i == 0) ky = 0; if (i == 1) ky = 0; }
                    if (j == 1)  { if (i == -1) ky = 1; if (i == 0) ky = 2; if (i == 1) ky = 1; }
                    gx += kx * lum;
                    gy += ky * lum;
                }
            }
            double mag = std::sqrt(gx * gx + gy * gy);
            unsigned char v = static_cast<unsigned char>(std::min(255.0, mag));
            int idxOut = (y * width + x) * channels;
            h_out[idxOut + 0] = v;
            h_out[idxOut + 1] = v;
            h_out[idxOut + 2] = v;
            if (channels == 4) h_out[idxOut + 3] = h_in[idxOut + 3];
        }
    }
}

void cpuProcessor::cpuapplySharpen(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels) {
    if (!h_in || !h_out) return;
    // simple 3x3 sharpen kernel
    int kernel[3][3] = { {0, -1, 0}, {-1, 5, -1}, {0, -1, 0} };
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                int acc = 0;
                for (int j = -1; j <= 1; ++j) {
                    int ny = std::min(std::max(y + j, 0), height - 1);
                    for (int i = -1; i <= 1; ++i) {
                        int nx = std::min(std::max(x + i, 0), width - 1);
                        int idx = (ny * width + nx) * channels + c;
                        acc += kernel[j + 1][i + 1] * h_in[idx];
                    }
                }
                acc = std::min(255, std::max(0, acc));
                h_out[(y * width + x) * channels + c] = static_cast<unsigned char>(acc);
            }
        }
    }
}