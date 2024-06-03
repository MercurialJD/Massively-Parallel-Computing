#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"


#define TILE_WIDTH 16
#define UNROLL_BATCH_SIZE 5000


__global__ void unroll_kernel(const float *X, float *X_unroll, const int Batch_start, const int Batch_end, const int Channel, const int Height, const int Width, const int K) {
    // Referenced from the textbook and lecture 12 slides
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int W_unroll = Height_out * Width_out;        // Width of unrolled matrix

    #define X_4d(i3, i2, i1, i0) X[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + (i0)]
    #define X_unroll(i2, i1, i0) X_unroll[(i2) * (Channel * K * K * Height_out * Width_out) + (i1) * (Height_out * Width_out) + (i0)]

    if (t < Channel * W_unroll && b < (Batch_end - Batch_start)) {
        int c = t / W_unroll;                           // Channel
        int w_unroll = t % W_unroll;                    // Linearized index of the output element

        int h_out = w_unroll / Width_out;
        int w_out = w_unroll % Width_out;

        int w_base = c * K * K;                         // Begining of the unrolled matrix for channel c
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < K; ++j)
                X_unroll(b, w_base + i * K + j, w_unroll) = X_4d(b + Batch_start, c, h_out + i, w_out + j);
    }

    #undef X_4d
    #undef X_unroll
}


__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns,
                                     int Batch_start, int Batch_end) {
    // From lab3

    __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

    unsigned int bx = blockIdx.x; unsigned int by = blockIdx.y; unsigned int bz = blockIdx.z;
    unsigned int tx = threadIdx.x; unsigned int ty = threadIdx.y; unsigned int tz = threadIdx.z;

    #define A_2d(i1, i0) A[(i1) * numAColumns + (i0)]
    #define B_3d(i2, i1, i0) B[(i2) * (numBColumns * numBRows) + (i1) * (numBColumns) + (i0)]
    #define C_3d(i2, i1, i0) C[(i2) * (numCColumns * numCRows) + (i1) * (numCColumns) + (i0)]

    unsigned int bat = bz * blockDim.z + tz;
    unsigned int row = by * blockDim.y + ty;
    unsigned int col = bx * blockDim.x + tx;

    float cVal = 0;
    for (size_t k = 0; k < (numAColumns - 1) / TILE_WIDTH + 1; ++k) {
        // Load partial A to shared memory
        if (row < numARows && (k * TILE_WIDTH + tx) < numAColumns)
            subTileA[ty][tx] = A_2d(row, k * TILE_WIDTH + tx);
        else
            subTileA[ty][tx] = 0.0f;

        // Load partial B to shared memory
        if ((k * TILE_WIDTH + ty) < numBRows && col < numBColumns)
            subTileB[ty][tx] = B_3d(bat, k * TILE_WIDTH + ty, col);
        else
            subTileB[ty][tx] = 0.0f;

        __syncthreads();

        // Sum per tile
        if (row < numCRows && col < numCColumns)
            for (size_t i = 0; i < TILE_WIDTH; ++i)
                cVal += subTileA[ty][i] * subTileB[i][tx];

        __syncthreads();
    }

    // Write to C if address is valid
    if (row < numCRows && col < numCColumns && bat < (Batch_end - Batch_start))
        C_3d(bat + Batch_start, row, col) = cVal;

    #undef A_2d
    #undef B_3d
    #undef C_3d
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

    // Partition input into smaller batches to avoid memory overflow
    float *device_unrolled_input;
    cudaMalloc((void **) &device_unrolled_input, \
                        UNROLL_BATCH_SIZE * Channel * K * K * Height_out * Width_out * sizeof(float));

    for (int b = 0; b < Batch; b += UNROLL_BATCH_SIZE) {
        int curr_batch_size = min(Batch - b, UNROLL_BATCH_SIZE);

        // Unroll
        // K * K is not part of Grid/Block, they are iterated inside the kernel
        dim3 unrollGrid( ceil(1.0 * curr_batch_size / TILE_WIDTH), \
                            ceil(1.0 * Channel * Height_out * Width_out / TILE_WIDTH) );
        dim3 unrollBlock(TILE_WIDTH, TILE_WIDTH);
        unroll_kernel<<<unrollGrid, unrollBlock>>>(device_input, device_unrolled_input, \
                                                        b, b + curr_batch_size, Channel, Height, Width, K);

        // GEMM
        int mask_rows = Map_out;        int mask_cols = Channel * K * K;
        int X_rows = Channel * K * K;   int X_cols = Height_out * Width_out;
        int out_rows = mask_rows;       int out_cols = X_cols;

        dim3 gemmGrid( ceil(1.0 * out_cols / TILE_WIDTH), \
                        ceil(1.0 * out_rows / TILE_WIDTH), curr_batch_size );
        dim3 gemmBlock(TILE_WIDTH, TILE_WIDTH);
        matrixMultiplyShared<<<gemmGrid, gemmBlock>>>(device_mask, device_unrolled_input, device_output, \
                                        mask_rows, mask_cols, X_rows,X_cols, out_rows, out_cols, b, b + curr_batch_size);
    }

    cudaFree(device_unrolled_input);
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