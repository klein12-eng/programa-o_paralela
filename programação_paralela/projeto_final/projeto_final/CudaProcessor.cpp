#include "CudaProcessor.h"
#include <cstring>
#include <algorithm>

#ifndef HAVE_CUDA

// Fallback implementations when CUDA is not available on the build machine.
// These ensure the linker finds the symbols and provide CPU equivalents.

CudaProcessor::CudaProcessor() {}
CudaProcessor::~CudaProcessor() {}

void CudaProcessor::applyBlur(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius) {
    // Simple copy (no blur) fallback
    size_t img_size = static_cast<size_t>(width) * height * channels;
    std::memcpy(h_out, h_in, img_size);
}

void CudaProcessor::applyBlur_a(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius) {
    applyBlur(h_in, h_out, width, height, channels, radius);
}

void CudaProcessor::applyBlur_b(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius) {
    applyBlur(h_in, h_out, width, height, channels, radius);
}

void CudaProcessor::applyMixedFilter(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels) {
    // Fallback: produce grayscale (BW) for whole image
    size_t pixels = static_cast<size_t>(width) * height;
    for (size_t i = 0; i < pixels; ++i) {
        int idx = (int)(i * channels);
        int b = h_in[idx + 0];
        int g = h_in[idx + 1];
        int r = h_in[idx + 2];
        int gray = (int)(0.299 * r + 0.587 * g + 0.114 * b);
        unsigned char v = (unsigned char)std::min(255, gray);
        h_out[idx + 0] = v;
        h_out[idx + 1] = v;
        h_out[idx + 2] = v;
        if (channels == 4) h_out[idx + 3] = h_in[idx + 3];
    }
}

void CudaProcessor::applySepia(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels) {
    size_t pixels = static_cast<size_t>(width) * height;
    for (size_t i = 0; i < pixels; ++i) {
        int idx = (int)(i * channels);
        int b = h_in[idx + 0];
        int g = h_in[idx + 1];
        int r = h_in[idx + 2];
        int tr = (int)std::min(255, (int)(0.393 * r + 0.769 * g + 0.189 * b));
        int tg = (int)std::min(255, (int)(0.349 * r + 0.686 * g + 0.168 * b));
        int tb = (int)std::min(255, (int)(0.272 * r + 0.534 * g + 0.131 * b));
        h_out[idx + 0] = (unsigned char)tb;
        h_out[idx + 1] = (unsigned char)tg;
        h_out[idx + 2] = (unsigned char)tr;
        if (channels == 4) h_out[idx + 3] = h_in[idx + 3];
    }
}

void CudaProcessor::applyInvert(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels) {
    size_t img_size = static_cast<size_t>(width) * height * channels;
    for (size_t i = 0; i < img_size; ++i) {
        h_out[i] = (unsigned char)(255 - h_in[i]);
    }
}

std::vector<int> CudaProcessor::calculateHistogram(const unsigned char* h_img, int width, int height, int channels, int channel_idx) {
    std::vector<int> histogram(256, 0);
    if (channel_idx >= channels) return histogram;
    size_t pixels = static_cast<size_t>(width) * height;
    for (size_t i = 0; i < pixels; ++i) {
        int val = h_img[i * channels + channel_idx];
        histogram[val]++;
    }
    return histogram;
}

void CudaProcessor::checkCudaError(cudaError_t err, const char* file, int line) {
    // noop in fallback
}

// Provide fallback isCudaAvailable when CUDA headers/driver not present
bool CudaProcessor::isCudaAvailable() {
    return false;
}

#endif // HAVE_CUDA
