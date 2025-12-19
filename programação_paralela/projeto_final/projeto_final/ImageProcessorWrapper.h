// ImageProcessorWrapper.h

#pragma once

#include <msclr/gcroot.h>
#include "CudaProcessor.h"
#include "cpuProcessor.h"

public ref class ImageProcessorWrapper {
public:
    // Constructor creates the native object
    ImageProcessorWrapper() {
        nativeCpuProcessor = new cpuProcessor();
#if defined(HAVE_CUDA)
        nativeCudaProcessor = new CudaProcessor();
#else
        nativeCudaProcessor = nullptr;
#endif
    }

    // Destructor cleans up the native object
    ~ImageProcessorWrapper() {
        this->!ImageProcessorWrapper();
    }

    // Expose CUDA availability to managed code
    bool IsCudaAvailable() {
        if (nativeCudaProcessor == nullptr) return false;
        return nativeCudaProcessor->isCudaAvailable();
    }

    // CPU implementations
    System::Drawing::Bitmap^ ApplyBW_CPU(System::Drawing::Bitmap^ inputBitmap) {
        int w,h,channels;
        unsigned char* inBuf = BitmapToBuffer(inputBitmap, w, h, channels);
        unsigned char* outBuf = new unsigned char[w * h * channels];
        nativeCpuProcessor->cpuapplyBW(inBuf, outBuf, w, h, channels);
        System::Drawing::Bitmap^ outBmp = BufferToBitmap(outBuf, w, h, channels);
        delete[] inBuf; delete[] outBuf;
        return outBmp;
    }

    System::Drawing::Bitmap^ ApplySepia_CPU(System::Drawing::Bitmap^ inputBitmap) {
        int w,h,channels;
        unsigned char* inBuf = BitmapToBuffer(inputBitmap, w, h, channels);
        unsigned char* outBuf = new unsigned char[w * h * channels];
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
        System::Drawing::Bitmap^ outBmp = BufferToBitmap(outBuf, w, h, channels);
        delete[] inBuf; delete[] outBuf;
        return outBmp;
    }

    System::Drawing::Bitmap^ ApplyInvert_CPU(System::Drawing::Bitmap^ inputBitmap) {
        int w,h,channels;
        unsigned char* inBuf = BitmapToBuffer(inputBitmap, w, h, channels);
        unsigned char* outBuf = new unsigned char[w * h * channels];
        for (int i = 0; i < w * h; ++i) {
            int idx = i * channels;
            outBuf[idx + 0] = 255 - inBuf[idx + 0];
            outBuf[idx + 1] = 255 - inBuf[idx + 1];
            outBuf[idx + 2] = 255 - inBuf[idx + 2];
            if (channels == 4) outBuf[idx + 3] = inBuf[idx + 3];
        }
        System::Drawing::Bitmap^ outBmp = BufferToBitmap(outBuf, w, h, channels);
        delete[] inBuf; delete[] outBuf;
        return outBmp;
    }

    System::Drawing::Bitmap^ ApplyGaussian_CPU(System::Drawing::Bitmap^ inputBitmap) {
        int w,h,channels;
        unsigned char* inBuf = BitmapToBuffer(inputBitmap, w, h, channels);
        unsigned char* outBuf = new unsigned char[w * h * channels];
        nativeCpuProcessor->cpuapplyGaussian(inBuf, outBuf, w, h, channels, 3, 1.0);
        System::Drawing::Bitmap^ outBmp = BufferToBitmap(outBuf, w, h, channels);
        delete[] inBuf; delete[] outBuf; return outBmp;
    }

    System::Drawing::Bitmap^ ApplyEdge_CPU(System::Drawing::Bitmap^ inputBitmap) {
        int w,h,channels;
        unsigned char* inBuf = BitmapToBuffer(inputBitmap, w, h, channels);
        unsigned char* outBuf = new unsigned char[w * h * channels];
        nativeCpuProcessor->cpuapplyEdge(inBuf, outBuf, w, h, channels);
        System::Drawing::Bitmap^ outBmp = BufferToBitmap(outBuf, w, h, channels);
        delete[] inBuf; delete[] outBuf; return outBmp;
    }

    System::Drawing::Bitmap^ ApplySharpen_CPU(System::Drawing::Bitmap^ inputBitmap) {
        int w,h,channels;
        unsigned char* inBuf = BitmapToBuffer(inputBitmap, w, h, channels);
        unsigned char* outBuf = new unsigned char[w * h * channels];
        nativeCpuProcessor->cpuapplySharpen(inBuf, outBuf, w, h, channels);
        System::Drawing::Bitmap^ outBmp = BufferToBitmap(outBuf, w, h, channels);
        delete[] inBuf; delete[] outBuf; return outBmp;
    }

    // CUDA implementations (fallback to CPU if CUDA not available)
    System::Drawing::Bitmap^ ApplyBW_CUDA(System::Drawing::Bitmap^ inputBitmap) {
#if defined(HAVE_CUDA)
        int w,h,channels;
        unsigned char* inBuf = BitmapToBuffer(inputBitmap, w, h, channels);
        unsigned char* outBuf = new unsigned char[w * h * channels];
        nativeCudaProcessor->applyMixedFilter(inBuf, outBuf, w, h, channels);
        System::Drawing::Bitmap^ outBmp = BufferToBitmap(outBuf, w, h, channels);
        delete[] inBuf; delete[] outBuf; return outBmp;
#else
        return ApplyBW_CPU(inputBitmap);
#endif
    }

    System::Drawing::Bitmap^ ApplySepia_CUDA(System::Drawing::Bitmap^ inputBitmap) {
#if defined(HAVE_CUDA)
        int w,h,channels;
        unsigned char* inBuf = BitmapToBuffer(inputBitmap, w, h, channels);
        unsigned char* outBuf = new unsigned char[w * h * channels];
        nativeCudaProcessor->applySepia(inBuf, outBuf, w, h, channels);
        System::Drawing::Bitmap^ outBmp = BufferToBitmap(outBuf, w, h, channels);
        delete[] inBuf; delete[] outBuf; return outBmp;
#else
        return ApplySepia_CPU(inputBitmap);
#endif
    }

    System::Drawing::Bitmap^ ApplyInvert_CUDA(System::Drawing::Bitmap^ inputBitmap) {
#if defined(HAVE_CUDA)
        int w,h,channels;
        unsigned char* inBuf = BitmapToBuffer(inputBitmap, w, h, channels);
        unsigned char* outBuf = new unsigned char[w * h * channels];
        nativeCudaProcessor->applyInvert(inBuf, outBuf, w, h, channels);
        System::Drawing::Bitmap^ outBmp = BufferToBitmap(outBuf, w, h, channels);
        delete[] inBuf; delete[] outBuf; return outBmp;
#else
        return ApplyInvert_CPU(inputBitmap);
#endif
    }

    System::Drawing::Bitmap^ ApplyGaussian_CUDA(System::Drawing::Bitmap^ inputBitmap) {
#if defined(HAVE_CUDA)
        int w,h,channels;
        unsigned char* inBuf = BitmapToBuffer(inputBitmap, w, h, channels);
        unsigned char* outBuf = new unsigned char[w * h * channels];
        nativeCudaProcessor->applyGaussian(inBuf, outBuf, w, h, channels);
        System::Drawing::Bitmap^ outBmp = BufferToBitmap(outBuf, w, h, channels);
        delete[] inBuf; delete[] outBuf; return outBmp;
#else
        return ApplyGaussian_CPU(inputBitmap);
#endif
    }

    System::Drawing::Bitmap^ ApplyEdge_CUDA(System::Drawing::Bitmap^ inputBitmap) {
#if defined(HAVE_CUDA)
        int w,h,channels;
        unsigned char* inBuf = BitmapToBuffer(inputBitmap, w, h, channels);
        unsigned char* outBuf = new unsigned char[w * h * channels];
        nativeCudaProcessor->applyEdge(inBuf, outBuf, w, h, channels);
        System::Drawing::Bitmap^ outBmp = BufferToBitmap(outBuf, w, h, channels);
        delete[] inBuf; delete[] outBuf; return outBmp;
#else
        return ApplyEdge_CPU(inputBitmap);
#endif
    }

    System::Drawing::Bitmap^ ApplySharpen_CUDA(System::Drawing::Bitmap^ inputBitmap) {
#if defined(HAVE_CUDA)
        int w,h,channels;
        unsigned char* inBuf = BitmapToBuffer(inputBitmap, w, h, channels);
        unsigned char* outBuf = new unsigned char[w * h * channels];
        nativeCudaProcessor->applySharpen(inBuf, outBuf, w, h, channels);
        System::Drawing::Bitmap^ outBmp = BufferToBitmap(outBuf, w, h, channels);
        delete[] inBuf; delete[] outBuf; return outBmp;
#else
        return ApplySharpen_CPU(inputBitmap);
#endif
    }

protected:
    // Finalizer for garbage collection
    !ImageProcessorWrapper() {
        if (nativeCpuProcessor) { delete nativeCpuProcessor; nativeCpuProcessor = nullptr; }
        if (nativeCudaProcessor) { delete nativeCudaProcessor; nativeCudaProcessor = nullptr; }
    }

private:
    // Private helper implementations (defined inline to ensure CLR metadata)
    CudaProcessor* nativeCudaProcessor; // Pointer to our native C++ CUDA object
    cpuProcessor* nativeCpuProcessor;   // Pointer to our native C++ CPU object

    // Helper prototypes
    static unsigned char* BitmapToBuffer(System::Drawing::Bitmap^ bmp, int& width, int& height, int& channels) {
        using namespace System::Drawing;
        using namespace System::Drawing::Imaging;
        width = bmp->Width;
        height = bmp->Height;
        PixelFormat pf = bmp->PixelFormat;
        channels = (pf == PixelFormat::Format24bppRgb) ? 3 : 4;

        BitmapData^ bd = bmp->LockBits(System::Drawing::Rectangle(0,0,width,height), ImageLockMode::ReadOnly, pf);
        int stride = bd->Stride;
        unsigned char* buffer = new unsigned char[width * height * channels];

        unsigned char* src = (unsigned char*)bd->Scan0.ToPointer();
        for (int y = 0; y < height; ++y) {
            unsigned char* row = src + y * stride;
            for (int x = 0; x < width; ++x) {
                int srcIdx = x * channels;
                int dstIdx = (y * width + x) * channels;
                buffer[dstIdx + 0] = row[srcIdx + 0];
                buffer[dstIdx + 1] = row[srcIdx + 1];
                buffer[dstIdx + 2] = row[srcIdx + 2];
                if (channels == 4) buffer[dstIdx + 3] = row[srcIdx + 3];
            }
        }
        bmp->UnlockBits(bd);
        return buffer;
    }

    static System::Drawing::Bitmap^ BufferToBitmap(unsigned char* buffer, int width, int height, int channels) {
        using namespace System::Drawing;
        using namespace System::Drawing::Imaging;
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

};