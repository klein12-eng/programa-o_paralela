// ImageProcessorWrapper.cpp
#include "ImageProcessorWrapper.h"
#include "CudaProcessor.h"
#include "cpuProcessor.h"

#include <cstring>

#pragma managed(push, on)

// Using declarations for .NET types
using namespace System::Drawing;
using namespace System::Drawing::Imaging;
using namespace System::Runtime::InteropServices;

ImageProcessorWrapper::ImageProcessorWrapper() {
    nativeCudaProcessor = nullptr;
    nativeCpuProcessor = new cpuProcessor();
#if defined(HAVE_CUDA)
    nativeCudaProcessor = new CudaProcessor();
#endif
}

ImageProcessorWrapper::~ImageProcessorWrapper() {
    if (nativeCpuProcessor) delete nativeCpuProcessor;
    if (nativeCudaProcessor) delete nativeCudaProcessor;
}

// Helper to convert Bitmap to raw buffer (BGR or BGRA)
static unsigned char* BitmapToBuffer(Bitmap^ bmp, int& width, int& height, int& channels) {
    width = bmp->Width;
    height = bmp->Height;
    PixelFormat pf = bmp->PixelFormat;
    channels = (pf == PixelFormat::Format24bppRgb) ? 3 : 4;

    BitmapData^ bd = bmp->LockBits(System::Drawing::Rectangle(0,0,width,height), ImageLockMode::ReadOnly, pf);
    int stride = bd->Stride;
    int bytes = abs(stride) * height;
    unsigned char* buffer = new unsigned char[width * height * channels];

    unsigned char* src = (unsigned char*)bd->Scan0.ToPointer();
    for (int y = 0; y < height; ++y) {
        unsigned char* row = src + y * stride;
        for (int x = 0; x < width; ++x) {
            int srcIdx = x * channels;
            int dstIdx = (y * width + x) * channels;
            // On Windows Bitmaps are BGR
            buffer[dstIdx + 0] = row[srcIdx + 0];
            buffer[dstIdx + 1] = row[srcIdx + 1];
            buffer[dstIdx + 2] = row[srcIdx + 2];
            if (channels == 4) buffer[dstIdx + 3] = row[srcIdx + 3];
        }
    }
    bmp->UnlockBits(bd);
    return buffer;
}

// Helper to create Bitmap from buffer
static Bitmap^ BufferToBitmap(unsigned char* buffer, int width, int height, int channels) {
    PixelFormat pf = (channels == 3) ? PixelFormat::Format24bppRgb : PixelFormat::Format32bppArgb;
    Bitmap^ bmp = gcnew Bitmap(width, height, pf);
    BitmapData^ bd = bmp->LockBits(System::Drawing::Rectangle(0,0,width,height), ImageLockMode::WriteOnly, pf);
    int stride = bd->Stride;
    unsigned char* dst = (unsigned char*)bd->Scan0.ToPointer();
    for (int y = 0; y < height; ++y) {
        unsigned char* row = dst + y * stride;
        for (int x = 0; x < width; ++x) {
            int dstIdx = x * channels;
            int srcIdx = (y * width + x) * channels;
            row[dstIdx + 0] = buffer[srcIdx + 0];
            row[dstIdx + 1] = buffer[srcIdx + 1];
            row[dstIdx + 2] = buffer[srcIdx + 2];
            if (channels == 4) row[dstIdx + 3] = buffer[srcIdx + 3];
        }
    }
    bmp->UnlockBits(bd);
    return bmp;
}

Bitmap^ ImageProcessorWrapper::ApplyBW_CPU(Bitmap^ inputBitmap) {
    int w, h, channels;
    unsigned char* inBuf = BitmapToBuffer(inputBitmap, w, h, channels);
    unsigned char* outBuf = new unsigned char[w * h * channels];
    nativeCpuProcessor->cpuapplyBW(inBuf, outBuf, w, h, channels);
    Bitmap^ outBmp = BufferToBitmap(outBuf, w, h, channels);
    delete[] inBuf; delete[] outBuf;
    return outBmp;
}

Bitmap^ ImageProcessorWrapper::ApplySepia_CPU(Bitmap^ inputBitmap) {
    int w, h, channels;
    unsigned char* inBuf = BitmapToBuffer(inputBitmap, w, h, channels);
    unsigned char* outBuf = new unsigned char[w * h * channels];
    // reuse BW kernel by applying sepia formula on CPU
    for (int i = 0; i < w * h; ++i) {
        int idx = i * channels;
        int b = inBuf[idx + 0];
        int g = inBuf[idx + 1];
        int r = inBuf[idx + 2];
        int tr = (int)std::min(255, (int)(0.393 * r + 0.769 * g + 0.189 * b));
        int tg = (int)std::min(255, (int)(0.349 * r + 0.686 * g + 0.168 * b));
        int tb = (int)std::min(255, (int)(0.272 * r + 0.534 * g + 0.131 * b));
        outBuf[idx + 0] = (unsigned char)tb;
        outBuf[idx + 1] = (unsigned char)tg;
        outBuf[idx + 2] = (unsigned char)tr;
        if (channels == 4) outBuf[idx + 3] = inBuf[idx + 3];
    }
    Bitmap^ outBmp = BufferToBitmap(outBuf, w, h, channels);
    delete[] inBuf; delete[] outBuf;
    return outBmp;
}

Bitmap^ ImageProcessorWrapper::ApplyInvert_CPU(Bitmap^ inputBitmap) {
    int w, h, channels;
    unsigned char* inBuf = BitmapToBuffer(inputBitmap, w, h, channels);
    unsigned char* outBuf = new unsigned char[w * h * channels];
    for (int i = 0; i < w * h; ++i) {
        int idx = i * channels;
        outBuf[idx + 0] = 255 - inBuf[idx + 0];
        outBuf[idx + 1] = 255 - inBuf[idx + 1];
        outBuf[idx + 2] = 255 - inBuf[idx + 2];
        if (channels == 4) outBuf[idx + 3] = inBuf[idx + 3];
    }
    Bitmap^ outBmp = BufferToBitmap(outBuf, w, h, channels);
    delete[] inBuf; delete[] outBuf;
    return outBmp;
}

Bitmap^ ImageProcessorWrapper::ApplyBW_CUDA(Bitmap^ inputBitmap) {
#if defined(HAVE_CUDA)
    int w, h, channels;
    unsigned char* inBuf = BitmapToBuffer(inputBitmap, w, h, channels);
    unsigned char* outBuf = new unsigned char[w * h * channels];
    nativeCudaProcessor->applyMixedFilter(inBuf, outBuf, w, h, channels); // mixed includes bw in part
    Bitmap^ outBmp = BufferToBitmap(outBuf, w, h, channels);
    delete[] inBuf; delete[] outBuf;
    return outBmp;
#else
    throw gcnew System::Exception("CUDA not available");
#endif
}

Bitmap^ ImageProcessorWrapper::ApplySepia_CUDA(Bitmap^ inputBitmap) {
#if defined(HAVE_CUDA)
    int w, h, channels;
    unsigned char* inBuf = BitmapToBuffer(inputBitmap, w, h, channels);
    unsigned char* outBuf = new unsigned char[w * h * channels];
    nativeCudaProcessor->applySepia(inBuf, outBuf, w, h, channels);
    Bitmap^ outBmp = BufferToBitmap(outBuf, w, h, channels);
    delete[] inBuf; delete[] outBuf;
    return outBmp;
#else
    throw gcnew System::Exception("CUDA not available");
#endif
}

Bitmap^ ImageProcessorWrapper::ApplyInvert_CUDA(Bitmap^ inputBitmap) {
#if defined(HAVE_CUDA)
    int w, h, channels;
    unsigned char* inBuf = BitmapToBuffer(inputBitmap, w, h, channels);
    unsigned char* outBuf = new unsigned char[w * h * channels];
    nativeCudaProcessor->applyInvert(inBuf, outBuf, w, h, channels);
    Bitmap^ outBmp = BufferToBitmap(outBuf, w, h, channels);
    delete[] inBuf; delete[] outBuf;
    return outBmp;
#else
    throw gcnew System::Exception("CUDA not available");
#endif
}

#pragma managed(pop)