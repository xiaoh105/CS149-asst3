#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"


// return GB/sec
float GBPerSec(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}


// This is the CUDA "kernel" function that is run on the GPU.  You
// know this because it is marked as a __global__ function.
__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall thread index from position of thread in current
    // block, and given the block we are in (in this example only a 1D
    // calculation is needed so the code only looks at the .x terms of
    // blockDim and threadIdx.
    int index = blockIdx.x * blockDim.x + threadIdx.x;


    // this check is necessary to make the code work for values of N
    // that are not a multiple of the thread block size (blockDim.x)
    if (index < N)
       result[index] = alpha * x[index] + y[index];
}


// saxpyCuda --
//
// This function is regular C code running on the CPU.  It allocates
// memory on the GPU using CUDA API functions, uses CUDA API functions
// to transfer data from the CPU's memory address space to GPU memory
// address space, and launches the CUDA kernel function on the GPU.
void saxpyCuda(int N, float alpha, float* xarray, float* yarray, float* resultarray) {

    // must read both input arrays (xarray and yarray) and write to
    // output array (resultarray)
    int totalBytes = sizeof(float) * 3 * N;

    // compute number of blocks and threads per block.  In this
    // application we've hardcoded thread blocks to contain 512 CUDA
    // threads.
    const int threadsPerBlock = 512;

    // Notice the round up here.  The code needs to compute the number
    // of threads blocks needed such that there is one thread per
    // element of the arrays.  This code is written to work for values
    // of N that are not multiples of threadPerBlock.
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // These are pointers that will be pointers to memory allocated
    // *one the GPU*.  You should allocate these pointers via
    // cudaMalloc.  You can access the resulting buffers from CUDA
    // device kernel code (see the kernel function saxpy_kernel()
    // above) but you cannot access the contents these buffers from
    // this thread. CPU threads cannot issue loads and stores from GPU
    // memory!
    float* device_x = nullptr;
    float* device_y = nullptr;
    float* device_result = nullptr;

    float *pinned_x = nullptr;
    float *pinned_y = nullptr;
    float *pinned_res = nullptr;

    cudaHostAlloc(&pinned_x, sizeof(float) * N, cudaHostAllocDefault);
    cudaHostAlloc(&pinned_y, sizeof(float) * N, cudaHostAllocDefault);
    cudaHostAlloc(&pinned_res, sizeof(float) * N, cudaHostAllocDefault);

    memcpy(pinned_x, xarray, sizeof(float) * N);
    memcpy(pinned_y, yarray, sizeof(float) * N);

    cudaMalloc(&device_x, sizeof(float) * N);
    cudaMalloc(&device_y, sizeof(float) * N);
    cudaMalloc(&device_result, sizeof(float) * N);
        
    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    cudaMemcpy(device_x, pinned_x, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, pinned_y, sizeof(float) * N, cudaMemcpyHostToDevice);

    double startRun = CycleTimer::currentSeconds();
   
    // run CUDA kernel. (notice the <<< >>> brackets indicating a CUDA
    // kernel launch) Execution on the GPU occurs here.
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
    cudaDeviceSynchronize();

    double endRun = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, device_result, sizeof(float) * N, cudaMemcpyDeviceToHost);

    double endTime = CycleTimer::currentSeconds();
    
    // end timing after result has been copied back into host memory

    memcpy(resultarray, pinned_res, sizeof(float) * N);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n",
		errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    double runDuration = endRun - startRun;
    printf("Effective BW by CUDA saxpy: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, GBPerSec(totalBytes, overallDuration - runDuration));
    printf("Run time: %.3f\n", 1000.f * runDuration);

    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);
    
}

void printCudaInfo() {

    // print out stats about the GPU in the machine.  Useful if
    // students want to know what GPU they are running on.

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
        printf("   Shared mem per block: %.0f KB\n", static_cast<float>(deviceProps.sharedMemPerBlock));
    }
    printf("---------------------------------------------------------\n");
}
