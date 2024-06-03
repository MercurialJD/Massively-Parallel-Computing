#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"


#define TILE_WIDTH 16
#define BLOCK_WIDTH ((TILE_WIDTH) + (K) - 1)


__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    extern __shared__ float shared_mem[];
    float* tile = shared_mem;
    float* mask_s = shared_mem + (BLOCK_WIDTH * BLOCK_WIDTH);

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    const int H_grid = ceil(1.0 * Height_out / TILE_WIDTH);
    const int W_grid = ceil(1.0 * Width_out / TILE_WIDTH);

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define tile_2d(i1, i0) tile[(i1) * (BLOCK_WIDTH) + (i0)]
    #define mask_2d(i1, i0) mask_s[(i1) * (K) + (i0)]

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x, ty = threadIdx.y;
    int h_start = (blockIdx.y / W_grid) * TILE_WIDTH, w_start = (blockIdx.y % W_grid) * TILE_WIDTH;

    int m = blockIdx.x;
    int h = h_start + ty;
    int w = w_start + tx;
    int b = blockIdx.z;

    float acc = 0.0f;
    for (int c = 0; c < Channel; ++c) {
        // Copy data to shared mem
        // Copy a channel of the mask
        if (ty < K && tx < K)
            mask_2d(ty, tx) = mask_4d(m, c, ty, tx);
        __syncthreads();

        // Copy multiple times to fill the tile block since we only
        // have TILE_WIDTH * TILE_WIDTH threads per block
        for (int i = ty; i < BLOCK_WIDTH; i += TILE_WIDTH) {
            for (int j = tx; j < BLOCK_WIDTH; j += TILE_WIDTH) {
                if ((h_start + i < Height) && (w_start + j < Width))
                    tile_2d(i, j) = in_4d(b, c, h_start + i, w_start + j);
                else
                    tile_2d(i, j) = 0.0f;
            }
        }
        __syncthreads();

        // Convolution
        for (int p = 0; p < K; ++p)
            for (int q = 0; q < K; ++q)
                acc += tile_2d(ty + p, tx + q) * mask_2d(p, q);
        __syncthreads();
    }

    if (h < Height_out && w < Width_out)
        out_4d(b, m, h, w) = acc;

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef tile_2d
    #undef mask_2d
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    int input_size = Batch * Channel * Height * Width * sizeof(float);
    int output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    int mask_size = Map_out * Channel * K * K * sizeof(float);

    cudaMalloc((void **) device_input_ptr, input_size);
    cudaMalloc((void **) device_output_ptr, output_size);
    cudaMalloc((void **) device_mask_ptr, mask_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    const int H_grid = ceil(1.0 * Height_out / TILE_WIDTH);
    const int W_grid = ceil(1.0 * Width_out / TILE_WIDTH);

    dim3 dimGrid(Map_out, H_grid * W_grid, Batch);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    int shared_mem_size = (BLOCK_WIDTH * BLOCK_WIDTH + K * K) * sizeof(float);

    // Need to include the size of shared memory as a third parameter
    // Otherwise error when try to allocate dynamic shared mem in kernel
    conv_forward_kernel<<<dimGrid, dimBlock, shared_mem_size>>>(device_output, device_input, device_mask, \
                                                    Batch, Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}