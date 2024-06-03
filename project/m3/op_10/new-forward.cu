#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include "gpu-new-forward.h"


#define TILE_WIDTH 16


__global__ void conv_forward_kernel(__half2 *output, const __half2 *input, const __half2 *mask,
                                        const int Batch, const int Map_out, const int Channel,
                                            const int Height, const int Width, const int K) {
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + (i0)]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + (i0)]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + (i0)]

    const int H_grid = ceil(1.0 * Height_out / TILE_WIDTH);
    const int W_grid = ceil(1.0 * Width_out / TILE_WIDTH);

    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    int b = blockIdx.z;

    if (h < Height_out && w < Width_out) {
        __half2 acc = __float2half2_rn(0.0f);

        for (int c = 0; c < Channel; ++c) {
            for (int p = 0; p < K; ++p) {
                for (int q = 0; q < K; ++q) {
                    // Load and compute for batches b and b + 1 as a group
                    __half2 input_h2 = in_4d(b, c, h + p, w + q);
                    __half2 mask_h2 = mask_4d(m, c, p, q);

                    acc = __hfma2(input_h2, mask_h2, acc);
                }
            }
        }

        out_4d(b, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
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

    int input_batch_dim = Channel * Height * Width;
    int input_dim = Batch * input_batch_dim;
    int output_dim = Batch * Map_out * (Height - K + 1) * (Width - K + 1);
    int mask_dim = Map_out * Channel * K * K;

    int half_input_size = input_dim * sizeof(float) / 2;
    int half_output_size = output_dim * sizeof(float) / 2;
    int mask_size = mask_dim * sizeof(float);

    cudaMalloc((void **) device_input_ptr, half_input_size);
    cudaMalloc((void **) device_output_ptr, half_output_size);
    cudaMalloc((void **) device_mask_ptr, mask_size);

    // Zip batch b and b + 1 together
    __half2 *half_host_input = new __half2[input_dim / 2];
    for (size_t b = 0; b < Batch; b += 2) {
        for (size_t i = 0; i < input_batch_dim; ++i)
            half_host_input[b / 2 * input_batch_dim + i] = __floats2half2_rn( host_input[b * input_batch_dim + i],
                                                                              host_input[(b + 1) * input_batch_dim + i] );
    }

    // Repeat each mask value twice
    __half2 *half_host_mask = new __half2[mask_dim];
    for (size_t i = 0; i < mask_dim; ++i)
        half_host_mask[i] = __float2half2_rn( host_mask[i] );

    cudaMemcpy(*device_input_ptr, half_host_input, half_input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, half_host_mask, mask_size, cudaMemcpyHostToDevice);

    delete[] half_host_input;
    delete[] half_host_mask;
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    const int H_grid = ceil(1.0 * Height_out / TILE_WIDTH);
    const int W_grid = ceil(1.0 * Width_out / TILE_WIDTH);

    dim3 dimGrid(Map_out, H_grid * W_grid, Batch / 2);          // Consecutive batches will be processed together
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    conv_forward_kernel<<<dimGrid, dimBlock>>>((__half2 *) device_output, (__half2 *) device_input, (__half2 *) device_mask, \
                                                    Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int batch_dim = Map_out * (Height - K + 1) * (Width - K + 1);
    int output_dim = Batch * batch_dim;

    __half2 *half_host_output = new __half2[output_dim / 2];
    cudaMemcpy(half_host_output, device_output, output_dim * sizeof(float) / 2, cudaMemcpyDeviceToHost);

    // Uznip batch b and b + 1 to host_output
    for (size_t b = 0; b < Batch; b += 2) {
        for (size_t i = 0; i < batch_dim; ++i) {
            float2 output_f = __half22float2( half_host_output[b / 2 * batch_dim + i] );
            host_output[b * batch_dim + i] = output_f.x;
            host_output[(b + 1) * batch_dim + i] = output_f.y;
        }
    }

    delete[] half_host_output;

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