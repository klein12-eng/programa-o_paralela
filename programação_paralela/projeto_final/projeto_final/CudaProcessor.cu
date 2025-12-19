// CudaProcessor.cu
#include "CudaProcessor.h"
#include <iostream>
#include <vector>
#if defined(HAVE_CUDA)
#include <cuda_runtime.h>
#endif

// =================================================================================
// CUDA KERNELS
// These functions run in parallel on the GPU.
// =================================================================================

#if defined(HAVE_CUDA)
__global__ void blurKernel(const unsigned char* in, unsigned char* out, int width, int height, int channels, int radius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        float r_acc = 0.0f, g_acc = 0.0f, b_acc = 0.0f;
        int pixel_count = 0;
        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                int current_row = row + y;
                int current_col = col + x;
                if (current_row >= 0 && current_row < height && current_col >= 0 && current_col < width) {
                    int idx = (current_row * width + current_col) * channels;
                    r_acc += in[idx + 0];
                    g_acc += in[idx + 1];
                    b_acc += in[idx + 2];
                    pixel_count++;
                }
            }
        }
        int out_idx = (row * width + col) * channels;
        out[out_idx + 0] = static_cast<unsigned char>(r_acc / pixel_count);
        out[out_idx + 1] = static_cast<unsigned char>(g_acc / pixel_count);
        out[out_idx + 2] = static_cast<unsigned char>(b_acc / pixel_count);
        if (channels == 4) out[out_idx + 3] = in[out_idx + 3];
    }
}

__global__ void mixedFilterKernel(const unsigned char* in, unsigned char* out, int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int idx = (row * width + col) * channels;
        float r_in = in[idx + 0];
        float g_in = in[idx + 1];
        float b_in = in[idx + 2];
        float r_out, g_out, b_out;
        if (col < width / 2 && row < height / 2) {
            r_out = 0.393f * r_in + 0.769f * g_in + 0.189f * b_in;
            g_out = 0.349f * r_in + 0.686f * g_in + 0.168f * b_in;
            b_out = 0.272f * r_in + 0.534f * g_in + 0.131f * b_in;
        }
        else if (col >= width / 2 && row >= height / 2) {
            float gray = 0.299f * r_in + 0.587f * g_in + 0.114f * b_in;
            r_out = gray; g_out = gray; b_out = gray;
        } else {
            r_out = r_in; g_out = g_in; b_out = b_in;
        }
        out[idx + 0] = static_cast<unsigned char>(min(255.0f, r_out));
        out[idx + 1] = static_cast<unsigned char>(min(255.0f, g_out));
        out[idx + 2] = static_cast<unsigned char>(min(255.0f, b_out));
        if (channels == 4) out[idx + 3] = in[idx + 3];
    }
}

__global__ void sepiaKernel(const unsigned char* in, unsigned char* out, int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int idx = (row * width + col) * channels;
        float r_in = in[idx + 0];
        float g_in = in[idx + 1];
        float b_in = in[idx + 2];
        float r = 0.393f * r_in + 0.769f * g_in + 0.189f * b_in;
        float g = 0.349f * r_in + 0.686f * g_in + 0.168f * b_in;
        float b = 0.272f * r_in + 0.534f * g_in + 0.131f * b_in;
        out[idx + 0] = static_cast<unsigned char>(min(255.0f, r));
        out[idx + 1] = static_cast<unsigned char>(min(255.0f, g));
        out[idx + 2] = static_cast<unsigned char>(min(255.0f, b));
        if (channels == 4) out[idx + 3] = in[idx + 3];
    }
}

__global__ void invertKernel(const unsigned char* in, unsigned char* out, int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int idx = (row * width + col) * channels;
        out[idx + 0] = 255 - in[idx + 0];
        out[idx + 1] = 255 - in[idx + 1];
        out[idx + 2] = 255 - in[idx + 2];
        if (channels == 4) out[idx + 3] = in[idx + 3];
    }
}

// Gaussian uses separable kernels; we implement a simple 1D convolution kernel for horizontal and vertical passes.
__global__ void gaussianHorizontal(const unsigned char* in, unsigned char* out, int width, int height, int channels, const double* k, int radius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;
    int idxBase = (row * width + col) * channels;
    for (int c = 0; c < channels; ++c) {
        double acc = 0.0;
        for (int i = -radius; i <= radius; ++i) {
            int nx = col + i;
            if (nx < 0) nx = 0; if (nx >= width) nx = width - 1;
            int idx = (row * width + nx) * channels + c;
            acc += k[i + radius] * in[idx];
        }
        out[idxBase + c] = static_cast<unsigned char>(min(255.0, max(0.0, acc)));
    }
}

__global__ void gaussianVertical(const unsigned char* in, unsigned char* out, int width, int height, int channels, const double* k, int radius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;
    int idxBase = (row * width + col) * channels;
    for (int c = 0; c < channels; ++c) {
        double acc = 0.0;
        for (int j = -radius; j <= radius; ++j) {
            int ny = row + j;
            if (ny < 0) ny = 0; if (ny >= height) ny = height - 1;
            int idx = (ny * width + col) * channels + c;
            acc += k[j + radius] * in[idx];
        }
        out[idxBase + c] = static_cast<unsigned char>(min(255.0, max(0.0, acc)));
    }
}

__global__ void sobelKernel(const unsigned char* in, unsigned char* out, int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;
    double gx = 0.0, gy = 0.0;
    for (int j = -1; j <= 1; ++j) {
        int ny = min(max(row + j, 0), height - 1);
        for (int i = -1; i <= 1; ++i) {
            int nx = min(max(col + i, 0), width - 1);
            int idx = (ny * width + nx) * channels;
            int b = in[idx + 0]; int g = in[idx + 1]; int r = in[idx + 2];
            double lum = 0.299 * r + 0.587 * g + 0.114 * b;
            int kx = 0; int ky = 0;
            if (j == -1) { if (i == -1) {kx=-1; ky=-1;} if (i==0) {kx=0; ky=-2;} if (i==1){kx=1; ky=-1;} }
            if (j == 0)  { if (i == -1) {kx=-2; ky=0;} if (i==0) {kx=0; ky=0;} if (i==1){kx=2; ky=0;} }
            if (j == 1)  { if (i == -1) {kx=-1; ky=1;} if (i==0) {kx=0; ky=2;} if (i==1){kx=1; ky=1;} }
            gx += kx * lum; gy += ky * lum;
        }
    }
    double mag = sqrt(gx*gx + gy*gy);
    unsigned char v = static_cast<unsigned char>(min(255.0, mag));
    int outIdx = (row * width + col) * channels;
    out[outIdx+0] = v; out[outIdx+1] = v; out[outIdx+2] = v;
    if (channels==4) out[outIdx+3] = in[outIdx+3];
}

__global__ void sharpenKernel(const unsigned char* in, unsigned char* out, int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;
    int kernel[3][3] = { {0, -1, 0}, {-1, 5, -1}, {0, -1, 0} };
    for (int c = 0; c < channels; ++c) {
        int acc = 0;
        for (int j = -1; j <= 1; ++j) {
            int ny = min(max(row + j, 0), height - 1);
            for (int i = -1; i <= 1; ++i) {
                int nx = min(max(col + i, 0), width - 1);
                int idx = (ny * width + nx) * channels + c;
                acc += kernel[j+1][i+1] * in[idx];
            }
        }
        int outIdx = (row * width + col) * channels + c;
        out[outIdx] = static_cast<unsigned char>(min(255, max(0, acc)));
    }
    if (channels==4) out[(row*width+col)*channels + 3] = in[(row*width+col)*channels + 3];
}
#endif // HAVE_CUDA

// =================================================================================
// CudaProcessor CLASS IMPLEMENTATION
// These functions run on the CPU and manage the CUDA operations.
// =================================================================================

CudaProcessor::CudaProcessor() {
#if defined(HAVE_CUDA)
    d_in_buffer = nullptr;
    d_out_buffer = nullptr;
    d_buffer_size = 0;
    cudaStreamCreate(&stream);
#endif
}

CudaProcessor::~CudaProcessor() {
#if defined(HAVE_CUDA)
    if (d_in_buffer) cudaFree(d_in_buffer);
    if (d_out_buffer) cudaFree(d_out_buffer);
    if (stream) cudaStreamDestroy(stream);
    cudaDeviceReset();
#endif
}

void CudaProcessor::checkCudaError(cudaError_t err, const char* file, int line) {
#if defined(HAVE_CUDA)
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << file << " at line " << line
            << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
#endif
}

#define CUDA_CHECK(err) checkCudaError(err, __FILE__, __LINE__)

#if defined(HAVE_CUDA)
// Ensure device buffers are allocated and big enough
static void ensureDeviceBuffers(CudaProcessor* self, size_t required) {
    if (self->d_buffer_size >= required) return;
    // Free old
    if (self->d_in_buffer) cudaFree(self->d_in_buffer);
    if (self->d_out_buffer) cudaFree(self->d_out_buffer);
    // Allocate new
    cudaMalloc(&self->d_in_buffer, required);
    cudaMalloc(&self->d_out_buffer, required);
    self->d_buffer_size = required;
}
#endif

void CudaProcessor::applyBlur(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius) {
    size_t img_size = static_cast<size_t>(width) * height * channels;
#if defined(HAVE_CUDA)
    ensureDeviceBuffers(this, img_size);

    // Use pinned host memory for faster async copy
    unsigned char* h_pinned_in = nullptr;
    unsigned char* h_pinned_out = nullptr;
    cudaMallocHost(&h_pinned_in, img_size);
    cudaMallocHost(&h_pinned_out, img_size);
    memcpy(h_pinned_in, h_in, img_size);

    // Async copy host -> device
    CUDA_CHECK(cudaMemcpyAsync(d_in_buffer, h_pinned_in, img_size, cudaMemcpyHostToDevice, stream));

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    blurKernel<<<gridDim, blockDim, 0, stream>>>(d_in_buffer, d_out_buffer, width, height, channels, radius);
    CUDA_CHECK(cudaGetLastError());

    // Async copy device -> host
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_out, d_out_buffer, img_size, cudaMemcpyDeviceToHost, stream));

    // Wait for stream to finish
    CUDA_CHECK(cudaStreamSynchronize(stream));

    memcpy(h_out, h_pinned_out, img_size);
    cudaFreeHost(h_pinned_in);
    cudaFreeHost(h_pinned_out);
#else
    // Fallback CPU implementation
    size_t pixels = static_cast<size_t>(width) * height;
    for (size_t i = 0; i < pixels; ++i) {
        int row = i / width;
        int col = i % width;
        float r_acc = 0.0f, g_acc = 0.0f, b_acc = 0.0f;
        int pixel_count = 0;
        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                int current_row = row + y;
                int current_col = col + x;
                if (current_row >= 0 && current_row < height && current_col >= 0 && current_col < width) {
                    int idx = (current_row * width + current_col) * channels;
                    r_acc += h_in[idx + 0];
                    g_acc += h_in[idx + 1];
                    b_acc += h_in[idx + 2];
                    pixel_count++;
                }
            }
        }
        int out_idx = (row * width + col) * channels;
        h_out[out_idx + 0] = static_cast<unsigned char>(r_acc / pixel_count);
        h_out[out_idx + 1] = static_cast<unsigned char>(g_acc / pixel_count);
        h_out[out_idx + 2] = static_cast<unsigned char>(b_acc / pixel_count);
        if (channels == 4) h_out[out_idx + 3] = h_in[out_idx + 3];
    }
#endif
}

void CudaProcessor::applyMixedFilter(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels) {
    size_t img_size = static_cast<size_t>(width) * height * channels;
#if defined(HAVE_CUDA)
    ensureDeviceBuffers(this, img_size);
    unsigned char* h_pinned_in = nullptr; unsigned char* h_pinned_out = nullptr;
    cudaMallocHost(&h_pinned_in, img_size);
    cudaMallocHost(&h_pinned_out, img_size);
    memcpy(h_pinned_in, h_in, img_size);
    CUDA_CHECK(cudaMemcpyAsync(d_in_buffer, h_pinned_in, img_size, cudaMemcpyHostToDevice, stream));
    dim3 blockDim(16,16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    mixedFilterKernel<<<gridDim, blockDim, 0, stream>>>(d_in_buffer, d_out_buffer, width, height, channels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_out, d_out_buffer, img_size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    memcpy(h_out, h_pinned_out, img_size);
    cudaFreeHost(h_pinned_in); cudaFreeHost(h_pinned_out);
#else
    // Fallback CPU
    size_t pixels = static_cast<size_t>(width) * height;
    for (size_t i = 0; i < pixels; ++i) {
        int row = i / width; int col = i % width;
        int idx = (row * width + col) * channels;
        int r_in = h_in[idx + 0]; int g_in = h_in[idx + 1]; int b_in = h_in[idx + 2];
        int r_out,g_out,b_out;
        if (col < width/2 && row < height/2) {
            r_out = (int)std::min(255, (int)(0.393f*r_in + 0.769f*g_in + 0.189f*b_in));
            g_out = (int)std::min(255, (int)(0.349f*r_in + 0.686f*g_in + 0.168f*b_in));
            b_out = (int)std::min(255, (int)(0.272f*r_in + 0.534f*g_in + 0.131f*b_in));
        } else if (col >= width/2 && row >= height/2) {
            int gray = (int)(0.299f*r_in + 0.587f*g_in + 0.114f*b_in);
            r_out = g_out = b_out = gray;
        } else { r_out=r_in; g_out=g_in; b_out=b_in; }
        h_out[idx+0] = (unsigned char)r_out; h_out[idx+1] = (unsigned char)g_out; h_out[idx+2] = (unsigned char)b_out;
        if (channels==4) h_out[idx+3] = h_in[idx+3];
    }
#endif
}

void CudaProcessor::applySepia(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels) {
    size_t img_size = static_cast<size_t>(width) * height * channels;
#if defined(HAVE_CUDA)
    ensureDeviceBuffers(this, img_size);
    unsigned char* h_pinned_in = nullptr; unsigned char* h_pinned_out = nullptr;
    cudaMallocHost(&h_pinned_in, img_size); cudaMallocHost(&h_pinned_out, img_size);
    memcpy(h_pinned_in, h_in, img_size);
    CUDA_CHECK(cudaMemcpyAsync(d_in_buffer, h_pinned_in, img_size, cudaMemcpyHostToDevice, stream));
    dim3 blockDim(32,32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    sepiaKernel<<<gridDim, blockDim, 0, stream>>>(d_in_buffer, d_out_buffer, width, height, channels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_out, d_out_buffer, img_size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    memcpy(h_out, h_pinned_out, img_size);
    cudaFreeHost(h_pinned_in); cudaFreeHost(h_pinned_out);
#else
    size_t pixels = static_cast<size_t>(width) * height;
    for (size_t i = 0; i < pixels; ++i) {
        int idx = i * channels;
        int b = h_in[idx + 0]; int g = h_in[idx + 1]; int r = h_in[idx + 2];
        int tr = (int)std::min(255, (int)(0.393 * r + 0.769 * g + 0.189 * b));
        int tg = (int)std::min(255, (int)(0.349 * r + 0.686 * g + 0.168 * b));
        int tb = (int)std::min(255, (int)(0.272 * r + 0.534 * g + 0.131 * b));
        h_out[idx + 0] = (unsigned char)tb; h_out[idx + 1] = (unsigned char)tg; h_out[idx + 2] = (unsigned char)tr;
        if (channels == 4) h_out[idx + 3] = h_in[idx + 3];
    }
#endif
}

void CudaProcessor::applyInvert(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels) {
    size_t img_size = static_cast<size_t>(width) * height * channels;
#if defined(HAVE_CUDA)
    ensureDeviceBuffers(this, img_size);
    unsigned char* h_pinned_in = nullptr; unsigned char* h_pinned_out = nullptr;
    cudaMallocHost(&h_pinned_in, img_size); cudaMallocHost(&h_pinned_out, img_size);
    memcpy(h_pinned_in, h_in, img_size);
    CUDA_CHECK(cudaMemcpyAsync(d_in_buffer, h_pinned_in, img_size, cudaMemcpyHostToDevice, stream));
    dim3 blockDim(32,32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    invertKernel<<<gridDim, blockDim, 0, stream>>>(d_in_buffer, d_out_buffer, width, height, channels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_out, d_out_buffer, img_size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    memcpy(h_out, h_pinned_out, img_size);
    cudaFreeHost(h_pinned_in); cudaFreeHost(h_pinned_out);
#else
    size_t img_sz = static_cast<size_t>(width) * height * channels;
    for (size_t i = 0; i < img_sz; ++i) h_out[i] = 255 - h_in[i];
#endif
}

void CudaProcessor::applyBlur_a(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius) {
    applyBlur(h_in, h_out, width, height, channels, radius);
}

void CudaProcessor::applyBlur_b(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius) {
    applyBlur(h_in, h_out, width, height, channels, radius);
}

// This is a CPU-based function. It does not use CUDA.
std::vector<int> CudaProcessor::calculateHistogram(const unsigned char* h_img, int width, int height, int channels, int channel_idx) {
    std::vector<int> histogram(256, 0);
    if (channel_idx >= channels) {
        return histogram; // Return empty if channel is invalid
    }

    for (int i = 0; i < width * height; ++i) {
        int value = h_img[i * channels + channel_idx];
        histogram[value]++;
    }
    return histogram;
}

bool CudaProcessor::isCudaAvailable() {
#if defined(HAVE_CUDA)
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) return false;
    return true;
#else
    return false;
#endif
}

void CudaProcessor::applyGaussian(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels) {
    size_t img_size = static_cast<size_t>(width) * height * channels;
#if defined(HAVE_CUDA)
    // simple fixed radius/sigma
    int radius = 3; double sigma = 1.0;
    ensureDeviceBuffers(this, img_size);
    // Build kernel on host
    int ksize = 2*radius + 1;
    std::vector<double> k(ksize);
    double sum = 0.0; double s2 = 2.0 * sigma * sigma;
    for (int i = -radius; i <= radius; ++i) { double v = exp(-(i*i)/s2); k[i+radius]=v; sum+=v; }
    for (int i=0;i<ksize;++i) k[i]/=sum;

    // allocate device kernel
    double* d_k = nullptr; cudaMalloc(&d_k, ksize * sizeof(double)); cudaMemcpy(d_k, k.data(), ksize*sizeof(double), cudaMemcpyHostToDevice);

    unsigned char* h_pinned_in = nullptr; unsigned char* h_pinned_tmp = nullptr; unsigned char* h_pinned_out = nullptr;
    cudaMallocHost(&h_pinned_in, img_size); cudaMallocHost(&h_pinned_tmp, img_size); cudaMallocHost(&h_pinned_out, img_size);
    memcpy(h_pinned_in, h_in, img_size);
    CUDA_CHECK(cudaMemcpyAsync(d_in_buffer, h_pinned_in, img_size, cudaMemcpyHostToDevice, stream));

    dim3 block(16,16);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);
    gaussianHorizontal<<<grid, block, 0, stream>>>(d_in_buffer, d_out_buffer, width, height, channels, d_k, radius);
    CUDA_CHECK(cudaGetLastError());
    // swap buffers
    unsigned char* tmpbuf = d_in_buffer; d_in_buffer = d_out_buffer; d_out_buffer = tmpbuf;
    gaussianVertical<<<grid, block, 0, stream>>>(d_in_buffer, d_out_buffer, width, height, channels, d_k, radius);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_pinned_out, d_out_buffer, img_size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    memcpy(h_out, h_pinned_out, img_size);

    cudaFreeHost(h_pinned_in); cudaFreeHost(h_pinned_tmp); cudaFreeHost(h_pinned_out);
    cudaFree(d_k);
#else
    // fallback to CPU implementation using simple separable gaussian
    // reuse cpuProcessor style implementation inline to avoid adding dependency
    int radius = 3; double sigma = 1.0;
    // build kernel
    int ksize = 2*radius + 1; std::vector<double> k(ksize); double sum=0.0; double s2=2.0*sigma*sigma;
    for (int i=-radius;i<=radius;++i){ double v=exp(-(i*i)/s2); k[i+radius]=v; sum+=v; } for (int i=0;i<ksize;++i) k[i]/=sum;
    std::vector<unsigned char> tmp(img_size);
    // horizontal
    for (int y=0;y<height;++y) for (int x=0;x<width;++x) for (int c=0;c<channels;++c){ double acc=0.0; for(int i=-radius;i<=radius;++i){ int nx=x+i; if(nx<0) nx=0; if(nx>=width) nx=width-1; acc += k[i+radius]*h_in[(y*width+nx)*channels + c]; } tmp[(y*width+x)*channels + c] = (unsigned char)std::min(255.0, std::max(0.0, acc)); }
    // vertical
    for (int y=0;y<height;++y) for (int x=0;x<width;++x) for (int c=0;c<channels;++c){ double acc=0.0; for(int j=-radius;j<=radius;++j){ int ny=y+j; if(ny<0) ny=0; if(ny>=height) ny=height-1; acc+= k[j+radius]* tmp[(ny*width+x)*channels + c]; } h_out[(y*width+x)*channels + c] = (unsigned char)std::min(255.0, std::max(0.0, acc)); }
#endif
}

void CudaProcessor::applyEdge(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels) {
    size_t img_size = static_cast<size_t>(width) * height * channels;
#if defined(HAVE_CUDA)
    ensureDeviceBuffers(this, img_size);
    unsigned char* h_pinned_in = nullptr; unsigned char* h_pinned_out = nullptr;
    cudaMallocHost(&h_pinned_in, img_size); cudaMallocHost(&h_pinned_out, img_size);
    memcpy(h_pinned_in, h_in, img_size);
    CUDA_CHECK(cudaMemcpyAsync(d_in_buffer, h_pinned_in, img_size, cudaMemcpyHostToDevice, stream));
    dim3 block(16,16); dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);
    sobelKernel<<<grid, block, 0, stream>>>(d_in_buffer, d_out_buffer, width, height, channels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_out, d_out_buffer, img_size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    memcpy(h_out, h_pinned_out, img_size);
    cudaFreeHost(h_pinned_in); cudaFreeHost(h_pinned_out);
#else
    // CPU fallback: simple Sobel
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
            double mag = sqrt(gx*gx + gy*gy);
            unsigned char v = static_cast<unsigned char>(std::min(255.0, mag));
            int outIdx = (y * width + x) * channels;
            h_out[outIdx+0] = v; h_out[outIdx+1] = v; h_out[outIdx+2] = v;
            if (channels==4) h_out[outIdx+3] = h_in[outIdx+3];
        }
    }
#endif
}

void CudaProcessor::applySharpen(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels) {
    size_t img_size = static_cast<size_t>(width) * height * channels;
#if defined(HAVE_CUDA)
    ensureDeviceBuffers(this, img_size);
    unsigned char* h_pinned_in = nullptr; unsigned char* h_pinned_out = nullptr;
    cudaMallocHost(&h_pinned_in, img_size); cudaMallocHost(&h_pinned_out, img_size);
    memcpy(h_pinned_in, h_in, img_size);
    CUDA_CHECK(cudaMemcpyAsync(d_in_buffer, h_pinned_in, img_size, cudaMemcpyHostToDevice, stream));
    dim3 block(16,16); dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);
    sharpenKernel<<<grid, block, 0, stream>>>(d_in_buffer, d_out_buffer, width, height, channels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_out, d_out_buffer, img_size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    memcpy(h_out, h_pinned_out, img_size);
    cudaFreeHost(h_pinned_in); cudaFreeHost(h_pinned_out);
#else
    // CPU fallback: simple 3x3 sharpen
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
#endif
}