
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
__device__ __host__
float4 normalize_quaternion(float4 q)
{
    float m = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
    q.x /= m;
    q.y /= m;
    q.z /= m;
    q.w /= m;
    return q;
}



__device__ __host__
float trilinear_interp(float* grid, float3 grid_lower, int3 grid_dims, 
	float3 grid_steps, int data_dims, int data_dim, float4 point)
{
	float px = point.x;
	float py = point.y;
	float pz = point.z;
	float pi = (px - grid_lower.x)/grid_steps.x - 0.5;
	float pj = (py - grid_lower.y)/grid_steps.y - 0.5;
	float pk = (pz - grid_lower.z)/grid_steps.z - 0.5;
	float ii = pi - floorf(pi);
	float jj = pj - floorf(pj);
	float kk = pk - floorf(pk);
	float dd = 0;
	for(int di = 0; di < 2; ++di)
	{
		for(int dj = 0; dj < 2; ++dj)
		{
			for(int dk = 0; dk < 2; ++dk)
			{
				int ci = (int)fmaxf(0, fminf(grid_dims.x - 1, floorf(pi + di)));
				int cj = (int)fmaxf(0, fminf(grid_dims.y - 1, floorf(pj + dj)));
				int ck = (int)fmaxf(0, fminf(grid_dims.z - 1, floorf(pk + dk)));
				float v = grid[ci*grid_dims.y*grid_dims.z*data_dims + 
							   cj*grid_dims.z*data_dims + 
							   ck*data_dims + 
							   data_dim];
				dd += v*(di ? ii : 1 - ii)*(dj ? jj : 1 - jj)*(dk ? kk : 1 - kk);
			}
		}
	}
	return dd;
}

__global__
void kernel_cudaFloatMemset(float* ptr, float value, int nvals)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < nvals; i += stride)
	{
		ptr[i] = value;
	}
}
void cudaFloatMemset(float* ptr, float value, int nvals, cudaStream_t stream)
{
	int nops = nvals/256;
    int numBlocks = ceil(nops * (1.0/256));
    dim3 blocks(numBlocks);
    dim3 threads(256);
    kernel_cudaFloatMemset<<<blocks, threads, 0, stream>>>(ptr, value, nvals);
}

__device__ 
float atomicMin(float *addr, float value)
{
    float old = *addr;
    float assumed;
    do
    {
    	if(old <= value) 
    		return old;
        assumed = old;
        old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));
    }while(old != assumed);
    return old;
}


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

