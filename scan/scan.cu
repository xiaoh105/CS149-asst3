#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256
#define MAX_THREADS_PER_BLOCK 1024

// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void upSweep(int *result, int n, int step) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((idx + 1) * step - 1 < n) {
        result[(idx + 1) * step - 1] += result[idx * step + (step >> 1) - 1];
    }
}

__global__ void downSweep(int *result, int n, int step) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long id1 = idx * step + (step >> 1) - 1;
    long long id2 = (idx + 1) * step - 1;
    if (id2 < n) {
        int tmp = result[id1];
        result[id1] = result[id2];
        result[id2] += tmp;
    }
}

void exclusiveScanNaive(int N, int* result)
{
    int rounded_n = nextPow2(N);
    for (int step = 1; step <= (rounded_n >> 1); step <<= 1) {
        int cur_step = step << 1;
        int thread_num = rounded_n / cur_step;
        int block_num = (thread_num - 1) / THREADS_PER_BLOCK + 1;
        upSweep<<<block_num, THREADS_PER_BLOCK>>>(result, rounded_n, cur_step);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("kernel launch failed with error: %s\n", cudaGetErrorString(err));
        }
    }
    int tmp = 0;
    cudaMemcpy(&result[rounded_n - 1], &tmp, sizeof(int), cudaMemcpyHostToDevice);
    for (int step = (rounded_n >> 1); step >= 1; step >>= 1) {
        int cur_step = step << 1;
        int thread_num = rounded_n / cur_step;
        int block_num = (thread_num - 1) / THREADS_PER_BLOCK + 1;
        downSweep<<<block_num, THREADS_PER_BLOCK>>>(result, rounded_n, cur_step);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("kernel launch failed with error: %s\n", cudaGetErrorString(err));
        }
    }
}

__global__ void scanBlock(int *result, int n, int *last) {
    extern __shared__ int output[MAX_THREADS_PER_BLOCK];
    extern __shared__ int sum[MAX_THREADS_PER_BLOCK / 32];
    int id = threadIdx.x;
    long long global_id = blockIdx.x * blockDim.x + threadIdx.x;
    n -= blockIdx.x * blockDim.x;
    // Note that scan with length <= 2 may cause array out of bound!!!
    if (id * 4 < min(n, MAX_THREADS_PER_BLOCK)) {
        reinterpret_cast<int4*>(output)[id] = 
            reinterpret_cast<int4*>(&result[blockIdx.x * blockDim.x])[threadIdx.x];
    }
    __syncthreads();
    int lane = id & 31;
    int warp_id = id >> 5;
    if (id < n) {
        auto mask = __activemask();
        if (lane >= 1) { output[id] += output[id - 1]; }
        __syncwarp(mask);
        if (lane >= 2) { output[id] += output[id - 2]; }
        __syncwarp(mask);
        if (lane >= 4) { output[id] += output[id - 4]; }
        __syncwarp(mask);
        if (lane >= 8) { output[id] += output[id - 8]; }
        __syncwarp(mask);
        if (lane >= 16) { output[id] += output[id - 16]; }
        __syncwarp(mask);
        if (lane == 31) { sum[warp_id] = output[id]; }
    }
    __syncthreads();
    if (warp_id == 0) {
        auto mask = __activemask();
        if (lane >= 1) { sum[lane] += sum[lane - 1]; }
        __syncwarp(mask);
        if (lane >= 2) { sum[lane] += sum[lane - 2]; }
        __syncwarp(mask);
        if (lane >= 4) { sum[lane] += sum[lane - 4]; }
        __syncwarp(mask);
        if (lane >= 8) { sum[lane] += sum[lane - 8]; }
        __syncwarp(mask);
        if (lane >= 16) { sum[lane] += sum[lane - 16]; }
        __syncwarp(mask);
        sum[lane] = (lane > 0) ? sum[lane - 1] : 0;
    }
    __syncthreads();
    if (id < n) {
        result[global_id] = (lane == 0 ? 0 : output[id - 1]) + sum[warp_id];
    }
    if (id == blockDim.x - 1 && last != nullptr) {
        last[blockIdx.x] = id < n ? output[id] + sum[warp_id] : 0;
    }
}

__global__ void addBlock(int *result, int n, int *last) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int block_id = blockIdx.x;
    if (id < n) {
        result[id] += last[block_id];
    }
}

void exclusiveScanBlock(int N, int *result) {
    int round_n = nextPow2(N);
    int block_num = (round_n - 1) / MAX_THREADS_PER_BLOCK + 1;
    int *last;
    int last_len = nextPow2((block_num - 1) / MAX_THREADS_PER_BLOCK + 1) * MAX_THREADS_PER_BLOCK;
    cudaMalloc(&last, last_len * sizeof(int));
    scanBlock<<<block_num, MAX_THREADS_PER_BLOCK>>>(result, round_n, last);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("kernel launch failed with error in stage 1: %s\n", cudaGetErrorString(err));
        exit(0);
    }
    if (block_num <= MAX_THREADS_PER_BLOCK) {
        scanBlock<<<1, MAX_THREADS_PER_BLOCK>>>(last, nextPow2(block_num), nullptr);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("kernel launch failed with error in stage 2: %s\n", cudaGetErrorString(err));
            exit(0);
        }
    } else {
        exclusiveScanBlock(block_num, last);
    }
    addBlock<<<block_num, MAX_THREADS_PER_BLOCK>>>(result, round_n, last);
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result)
{
    // exclusiveScanBlock(N, result);
    exclusiveScanBlock(N, result);
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}

__global__ void scanIdentical(int *input, int n, int *output) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id == 0) {
        output[0] = 0;
    }
    output[id + 1] = (id + 1 < n && input[id] == input[id + 1]) ? 1 : 0;
}

__global__ void collectOutput(int *input, int n, int *output) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id + 1 < n && input[id] < input[id + 1]) {
        output[input[id]] = id - 1;
    }
}

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {
    int round_n = nextPow2(length);
    int *tmp;
    cudaMalloc(&tmp, round_n * sizeof(int));
    int block_num = length / THREADS_PER_BLOCK + 1;
    scanIdentical<<<block_num, THREADS_PER_BLOCK>>>(device_input, length, device_output);
    cudaMemcpy(tmp, device_output, (length + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
    exclusiveScanBlock(length + 1, tmp);
    int res;
    cudaMemcpy(&res, &tmp[length], sizeof(int), cudaMemcpyDeviceToHost);
    collectOutput<<<block_num, THREADS_PER_BLOCK>>>(tmp, length + 1, device_output);
    cudaFree(tmp);
    return res;
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
        printf("   Shared mem per block: %.0f KB\n", static_cast<float>(deviceProps.sharedMemPerBlock));
        printf("   Max threads per block: %d cuda threads\n", deviceProps.maxThreadsPerBlock);
    }
    printf("---------------------------------------------------------\n"); 
}
