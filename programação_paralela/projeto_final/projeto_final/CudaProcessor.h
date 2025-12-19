// CudaProcessor.h

#pragma once
#include <vector>

#if defined(__has_include)
#  if __has_include(<cuda_runtime.h>)
#    include <cuda_runtime.h>
#    define HAVE_CUDA 1
#  endif
#endif

#ifndef HAVE_CUDA
// If CUDA headers aren't available, provide minimal fallback so headers compile.
typedef int cudaError_t;
#endif

// Forward declare the kernel to avoid including unnecessary headers in other translation units
struct cudaGraphicsResource;

class CudaProcessor {
public:
    CudaProcessor();
    ~CudaProcessor();

    // Processes an image from raw pixel data
    void applyBlur(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius);
    void applyBlur_a(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius);
    void applyBlur_b(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius);

    void applyMixedFilter(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels);
    void applySepia(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels);
    void applyInvert(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels);

    // New filters
    void applyGaussian(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels);
    void applyEdge(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels);
    void applySharpen(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels);

    // Calculates a histogram for one channel
    std::vector<int> calculateHistogram(const unsigned char* h_img, int width, int height, int channels, int channel_idx);

    // Return true if a real CUDA device is available and will be used
    bool isCudaAvailable();

private:
    // Helper for CUDA error checking
    void checkCudaError(cudaError_t err, const char* file, int line);

#if defined(HAVE_CUDA)
    // Persistent device buffers to avoid repeated cudaMalloc/cudaFree
    unsigned char* d_in_buffer = nullptr;
    unsigned char* d_out_buffer = nullptr;
    size_t d_buffer_size = 0; // in bytes
    cudaStream_t stream = 0;
#endif
};

#ifndef HAVE_CUDA
#include <cmath>
#include <algorithm>
// Provide simple inline CPU fallbacks so linker doesn't require CUDA .cu symbols
inline void CudaProcessor::applyGaussian(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels) {
    if (!h_in || !h_out) return;
    int radius = 3; double sigma = 1.0;
    int ksize = 2*radius + 1;
    std::vector<double> k(ksize);
    double sum = 0.0; double s2 = 2.0 * sigma * sigma;
    for (int i = -radius; i <= radius; ++i) { double v = std::exp(-(i*i)/s2); k[i+radius]=v; sum+=v; }
    for (int i=0;i<ksize;++i) k[i]/=sum;
    std::vector<unsigned char> tmp((size_t)width * height * channels);
    // horizontal
    for (int y=0;y<height;++y) for (int x=0;x<width;++x) for (int c=0;c<channels;++c){ double acc=0.0; for(int i=-radius;i<=radius;++i){ int nx=x+i; if(nx<0) nx=0; if(nx>=width) nx=width-1; acc += k[i+radius]*h_in[(y*width+nx)*channels + c]; } tmp[(y*width+x)*channels + c] = static_cast<unsigned char>(std::min(255.0, std::max(0.0, acc))); }
    // vertical
    for (int y=0;y<height;++y) for (int x=0;x<width;++x) for (int c=0;c<channels;++c){ double acc=0.0; for(int j=-radius;j<=radius;++j){ int ny=y+j; if(ny<0) ny=0; if(ny>=height) ny=height-1; acc+= k[j+radius]* tmp[(ny*width+x)*channels + c]; } h_out[(y*width+x)*channels + c] = static_cast<unsigned char>(std::min(255.0, std::max(0.0, acc))); }
}

inline void CudaProcessor::applyEdge(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels) {
    if (!h_in || !h_out) return;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double gx = 0.0, gy = 0.0;
            for (int j = -1; j <= 1; ++j) {
                int ny = std::min(std::max(y + j, 0), height - 1);
                for (int i = -1; i <= 1; ++i) {
                    int nx = std::min(std::max(x + i, 0), width - 1);
                    int idx = (ny * width + nx) * channels;
                    int b = h_in[idx + 0]; int g = h_in[idx + 1]; int r = h_in[idx + 2];
                    double lum = 0.299 * r + 0.587 * g + 0.114 * b;
                    int kx = 0; int ky = 0;
                    if (j == -1) { if (i == -1) {kx=-1; ky=-1;} if (i==0) {kx=0; ky=-2;} if (i==1){kx=1; ky=-1;} }
                    if (j == 0)  { if (i == -1) {kx=-2; ky=0;} if (i==0) {kx=0; ky=0;} if (i==1){kx=2; ky=0;} }
                    if (j == 1)  { if (i == -1) {kx=-1; ky=1;} if (i==0) {kx=0; ky=2;} if (i==1){kx=1; ky=1;} }
                    gx += kx * lum; gy += ky * lum;
                }
            }
            double mag = std::sqrt(gx*gx + gy*gy);
            unsigned char v = static_cast<unsigned char>(std::min(255.0, mag));
            int outIdx = (y * width + x) * channels;
            h_out[outIdx+0] = v; h_out[outIdx+1] = v; h_out[outIdx+2] = v;
            if (channels==4) h_out[outIdx+3] = h_in[outIdx+3];
        }
    }
}

inline void CudaProcessor::applySharpen(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels) {
    if (!h_in || !h_out) return;
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
#endif // HAVE_CUDA