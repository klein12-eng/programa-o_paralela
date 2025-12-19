//cpuProcessor.h

#pragma once


class cpuProcessor {
public:
    cpuProcessor();

    ~cpuProcessor();

    // Processes an image from raw pixel data
    void cpuapplyBlur(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius);
    void cpuapplyBlur_a(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius);
    void cpuapplyBlur_b(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius);
    void cpuapplyBW(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels);

    // New CPU filters
    void cpuapplyGaussian(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius, double sigma);
    void cpuapplyEdge(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels);
    void cpuapplySharpen(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels);

private:
    // Helper for CUDA error checking
  
};