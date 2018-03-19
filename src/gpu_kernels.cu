
#ifdef __cplusplus
extern "C" {
#endif

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define CUDA
#include "common_funcs.h"
#include "constants.h"
#include "gpu_kernels.h"

#define MAX_BLOCKS 65535




/* UTIL FUNCTIONS */


int PrintOnCudaError(const char* fn_name)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in %s: %s\n", fn_name, cudaGetErrorString(err));
        return 0;
    }
    return 1;
}


size_t GetSharedMemPerBlock(int device)
{
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, device);
    if(!PrintOnCudaError("GetSharedMemPerBlock"))
        return 0;
    else
        return p.sharedMemPerBlock;
}


/* Layer Funcs */

__global__
void kernel_convsp(
		const float* locs, 
		const float* data, 
		const float* weight, 
		const float* bias, 
		const int batch_size, 
		const int N, 
		const int nchannels, 
		const int ndims, 
		const int nkernels, 
		const int ncells, 
		const float radius, 
		const float* kernel_size, 
		const float* dilation, 
		const int dis_norm, 
		const int kernel_fn, 
		float* out, 
		float* ddata, 
		float* dweight, 
		const int block_size, 
		const int num_blocks)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N*batch_size*num_blocks; i += stride)
    {
    	int b = i/(N*num_blocks);
    	int n = (i % (N*num_blocks))/num_blocks;
    	int block = i % num_blocks;
    	int start = block*block_size;
    	compute_kernel_cells(locs, data, weight, bias, batch_size, N, 
    		nchannels, ndims, nkernels, ncells, radius, kernel_size, dilation, 
    		dis_norm, kernel_fn, out, b, n, start, start + block_size, ddata, dweight);
    }
}
int cuda_convsp(
		const float* locs, 
		const float* data, 
		const float* weight, 
		const float* bias, 
		const int batch_size, 
		const int N, 
		const int nchannels, 
		const int ndims, 
		const int nkernels, 
		const int ncells, 
		const float radius, 
		const float* kernel_size, 
		const float* dilation, 
		const int dis_norm, 
		const int kernel_fn, 
		float* out, 
		float* ddata, 
		float* dweight, 
		cudaStream_t stream, 
		const size_t nshared_device_mem)
{
	const int NUM_BLOCKS = 8;
	int block_size = ceilf(1.0f*N/NUM_BLOCKS);
	int nops = batch_size*N*NUM_BLOCKS;
    int numBlocks = ceil(nops * (1.0/256));
    dim3 blocks(numBlocks);
    dim3 threads(256);

	kernel_convsp<<<blocks, threads, 0, stream>>>(locs, data, weight, bias,
		batch_size, N, nchannels, ndims, nkernels, ncells, radius, kernel_size, 
		dilation, dis_norm, kernel_fn, out, ddata, dweight, block_size, NUM_BLOCKS);
	cudaDeviceSynchronize();
    return PrintOnCudaError("cuda_convsp");
}


__global__
void kernel_convsdf(
	const float* locs, 
	const int batch_size, 
	const int N, 
	const int ndims, 
	const float* idxs,
    const float* poses, 
    const float* scales, 
    const int M, 
    const int pose_len, 
    const float* sdfs, 
    const float* sdf_offsets, 
    const float* sdf_shapes, 
    const float* weight, 
    const float* bias, 
    const int nkernels, 
    const int ncells, 
    const float* kernel_size, 
    const float* dilation, 
    const float max_distance, 
    float* out, 
    float* dweight)
{
    int _isdf_cache[64];
    float _fsdf_cache[64];
    int* isdf_cache;
    float* fsdf_cache;
    if(M < 64)
    {
        isdf_cache = _isdf_cache;
        fsdf_cache = _fsdf_cache;
    }
    else
    {
        isdf_cache = (int*)malloc(sizeof(int)*M);
        fsdf_cache = (float*)malloc(sizeof(float)*M);
    }


    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N*batch_size*nkernels; i += stride)
    {
        int b = i/(N*nkernels);
        int n = (i % (N*nkernels))/nkernels;
        int outk = i % nkernels;

        compute_sdf_kernel_cells(locs, batch_size, N, ndims, idxs, poses, 
                    scales, M, pose_len, sdfs, sdf_offsets, sdf_shapes, weight, bias, 
                    nkernels, ncells, kernel_size, dilation, max_distance, out, b, n,
                    outk, dweight, isdf_cache, fsdf_cache);
    }

    if(M >= 64)
    {
        free(isdf_cache);
        free(fsdf_cache);
    }
}
int cuda_convsdf(
	const float* locs, 
	const int batch_size, 
	const int N, 
	const int ndims, 
	const float* idxs,
    const float* poses, 
    const float* scales, 
    const int M, 
    const int pose_len, 
    const float* sdfs, 
    const float* sdf_offsets, 
    const float* sdf_shapes, 
    const float* weight, 
    const float* bias, 
    const int nkernels, 
    const int ncells, 
    const float* kernel_size, 
    const float* dilation, 
    const float max_distance, 
    float* out, 
    float* dweight, 
    cudaStream_t stream)
{
    int nops = batch_size*N*nkernels;
    int numBlocks = ceil(nops * (1.0/256));
    dim3 blocks(numBlocks);
    dim3 threads(256); 

    // Stack overflow happens with the default stack size (1024).
    cudaError_t err = cudaDeviceSetLimit(cudaLimitStackSize, 4096);
    if (err != cudaSuccess) {
	    printf("error trying to set the stack size limit to 4096: %s\n", 
	    			cudaGetErrorString(err));
	        return 0;
    }

    kernel_convsdf<<<blocks, threads, 0, stream>>>(locs, batch_size, N, ndims, idxs, poses,
        scales, M, pose_len, sdfs, sdf_offsets, sdf_shapes, weight, bias, nkernels, ncells, 
        kernel_size, dilation, max_distance, out, dweight);
    cudaDeviceSynchronize();
    return PrintOnCudaError("cuda_convsdf");
}


#ifdef __cplusplus
}
#endif

