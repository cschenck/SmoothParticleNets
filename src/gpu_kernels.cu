
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
void kernel_convsp(float* locs, float* data, float* density, float* weight, float* bias, 
	int batch_size, int N, int nchannels, int ndims, int nkernels, int ncells, 
	float radius, float* kernel_size, float* dilation, float* out, float* ddata,
	float* dweight, int block_size, int num_blocks)
{
	extern __shared__ float shared_ptr[];

	// First compute where in the block we are.
	int b = blockIdx.x;
	int num_blocks2 = num_blocks*(num_blocks + 1)/2;
	int bi = num_blocks - 1 - floorf((sqrtf(8*(num_blocks2 - 1 - blockIdx.y) + 1) - 1)/2);
	int bj = num_blocks - num_blocks2 + blockIdx.y + 
				(num_blocks - 1 - bi)*(num_blocks - bi)/2;
	int starti = bi*block_size;
	int endi = fminf(N, (bi + 1)*block_size);
	int ni = endi - starti;
	int startj = bj*block_size;
	int endj = fminf(N, (bj + 1)*block_size);
	int nj = endj - startj;
	int th = threadIdx.x;
	int backwards = (ddata != NULL || dweight != NULL);
	int nweight_blocks = gridDim.z;
	int weight_block_size = ceilf(1.0f*nkernels/nweight_blocks);
	int bw = blockIdx.z;
	int startw = bw*weight_block_size;
	int endw = fminf(nkernels, (bw + 1)*weight_block_size);
	int nw = endw - startw;

	// Next copy over the weight.
	float* ptr = shared_ptr;
	int i, j;
	for(i = th; i < nw*nchannels*ncells; i += blockDim.x)
		ptr[i] = weight[i + startw*nchannels*ncells];
	float* weight_p = ptr;
	ptr += nw*nchannels*ncells;

	// Copy over the kernel_size and dilation.
	for(i = th; i < ndims; i += blockDim.x)
	{
		ptr[i] = kernel_size[i];
		ptr[ndims + i] = dilation[i];
	}
	float* kernel_size_p = ptr;
	float* dilation_p = ptr + ndims;
	ptr += 2*ndims;

	// Alocate space for dweight only if we're going backwards. Only copy
	// dweight over after.
	float* dweight_p = NULL;
	if(dweight != NULL)
	{
		for(i = th; i < nw*nchannels*ncells; i += blockDim.x)
			ptr[i] = 0.0f;
		dweight_p = ptr;
		ptr += nw*nchannels*ncells;
	}

	// Next we're going to copy over particle data. Since we're computing for 2 blocks,
	// make sure to copy the data over for each pointer sequentially.
	// Start with locs.
	float* locs_p = ptr;
	for(i = th; i < ni*(ndims + 1); i += blockDim.x)
		ptr[i] = locs[b*N*(ndims + 1) + starti*(ndims + 1) + i];
	ptr += ni*(ndims + 1);
	for(i = th; i < nj*(ndims + 1); i += blockDim.x)
		ptr[i] = locs[b*N*(ndims + 1) + startj*(ndims + 1) + i];
	ptr += nj*(ndims + 1);

	// Copy over data.
	float* data_p = ptr;
	for(i = th; i < ni*nchannels; i += blockDim.x)
		ptr[i] = data[b*N*nchannels + starti*nchannels + i];
	ptr += ni*nchannels;
	for(i = th; i < nj*nchannels; i += blockDim.x)
		ptr[i] = data[b*N*nchannels + startj*nchannels + i];
	ptr += nj*nchannels;

	// Copy over density.
	float* density_p = ptr;
	for(i = th; i < ni; i += blockDim.x)
		ptr[i] = density[b*N + starti + i];
	ptr += ni;
	for(i = th; i < nj; i += blockDim.x)
		ptr[i] = density[b*N + startj + i];
	ptr += nj;

	// Copy over out.
	float* out_p = ptr;
	for(i = th; i < ni; i += blockDim.x)
	{
		for(j = 0; j < nw; ++j)
			ptr[i*nw + j] = (backwards ? 
				out[b*N*nkernels + (starti + i)*nkernels + (startw + j)] : 0.0f);
	}
	ptr += ni*nw;
	for(i = th; i < nj; i += blockDim.x)
	{
		for(j = 0; j < nw; ++j)
			ptr[i*nw + j] = (backwards ? 
				out[b*N*nkernels + (startj + i)*nkernels + (startw + j)] : 0.0f);
	}
	ptr += nj*nw;

	// Alocate space for ddata only if we're going backwards. Only copy
	// ddata over after.
	float* ddata_p = NULL;
	if(ddata != NULL)
	{
		ddata_p = ptr;
		for(i = th; i < ni*nchannels; i += blockDim.x)
			ptr[i] = 0.0f;
		ptr += ni*nchannels;
		for(i = th; i < nj*nchannels; i += blockDim.x)
			ptr[i] = 0.0f;
		ptr += nj*nchannels;
	}

	// Wait for the copying to finish.
	__syncthreads();

	int nops = ceilf(1.0f*ni*nj/blockDim.x);
	for(i = th*nops; i < (th + 1)*nops;)
	{
		int n = i/nj;
		if(n >= ni) break;
		int start = (i % nj) + (bi == bj ? 0 : ni);
		int end = fminf(nj, (i % nj) + (th + 1)*nops - i) + (bi == bj ? 0 : ni);
		if(start >= ni + (bi == bj ? 0 : nj)) break;
		compute_kernel_cells(locs_p, data_p, density_p, weight_p, bias, 1, ni + nj, 
    		nchannels, ndims, nw, ncells, radius, kernel_size_p, dilation_p, 
    		out_p, 0, n, start, end, ddata_p, dweight_p);
		i += end - start;
	}

	// Wait again for everyone to finish before copying back over.
	__syncthreads();

	// If we're not going backwards, copy out back over.
	if(!backwards)
	{
		for(i = th; i < ni; i += blockDim.x)
		{
			for(j = 0; j < nw; ++j)
				atomicAdd(out + b*N*nkernels + (starti + i)*nkernels + (startw + j),
					out_p[i*nw + j]);
		}
		for(i = th; i < nj; i += blockDim.x)
		{
			for(j = 0; j < nw; ++j)
				atomicAdd(out + b*N*nkernels + (startj + i)*nkernels + (startw + j),
					out_p[i*nw + j + ni*nw]);
		}
	}

	// Copy ddata back over if it's not null.
	if(ddata != NULL)
	{
		for(i = th; i < ni*nchannels; i += blockDim.x)
			atomicAdd(ddata + b*N*nchannels + starti*nchannels + i, ddata_p[i]);
		for(i = th; i < nj*nchannels; i += blockDim.x)
			atomicAdd(ddata + b*N*nchannels + startj*nchannels + i, 
				ddata_p[i + ni*nchannels]);
	}

	// Copy dweight back over if it's not null.
	if(dweight != NULL)
	{
		for(i = th; i < nw*nchannels*ncells; i += blockDim.x)
			atomicAdd(dweight + startw*nchannels*ncells + i, dweight_p[i]);
	}
}
int cuda_convsp(float* locs, float* data, float* density, float* weight, float* bias, 
	int batch_size, int N, int nchannels, int ndims, int nkernels, int ncells, 
	float radius, float* kernel_size, float* dilation, float* out, float* ddata,
	float* dweight, cudaStream_t stream, size_t nshared_device_mem)
{
	size_t fixedmem = 0;
	int nweight_blocks;
	int nw;
	for(nweight_blocks = 1; nweight_blocks <= nkernels; ++nweight_blocks)
	{
		nw = ceilf(1.0f*nkernels/nweight_blocks);
		fixedmem = 0;
		fixedmem += nw*nchannels*ncells*sizeof(float); // weight
		fixedmem += ndims*sizeof(float); // kernel_size
		fixedmem += ndims*sizeof(float); // dilation
		if(dweight != NULL)
    		fixedmem += nw*nchannels*ncells*sizeof(float);
    	if(fixedmem <= nshared_device_mem/2)
    		break;
	}
	if(nweight_blocks > nkernels)
	{
		printf("error in cuda_convsp: Unable to fit weights in less than half of "
			"shared memory!\n");
		return 0;
	}
	size_t memperparticle = (ndims + 1)*sizeof(float) + // locs
							nchannels*sizeof(float) +   // data
							sizeof(float) +             // density
							nw*sizeof(float);           // out
    if(ddata != NULL)
    	memperparticle += nchannels*sizeof(float);
    

    int block_size = (nshared_device_mem - fixedmem)/(2*memperparticle);
    if(block_size > N) block_size = N;
    int nblocks = ceil(1.0f*N/block_size);
    size_t nshared_mem = fixedmem + 2*block_size*memperparticle;

    dim3 blocks(batch_size, nblocks*(nblocks + 1)/2, nweight_blocks);
    dim3 threads(min(256, block_size*block_size));

	kernel_convsp<<<blocks, threads, nshared_mem, stream>>>(locs, data, density, weight, bias,
		batch_size, N, nchannels, ndims, nkernels, ncells, radius, kernel_size, 
		dilation, out, ddata, dweight, block_size, nblocks);
	cudaDeviceSynchronize();
    return PrintOnCudaError("cuda_convsp");
}


__global__
void kernel_convsdf(float* locs, int batch_size, int N, int ndims, float* idxs,
    float* poses, float* scales, int M, int pose_len, float* sdfs, float* sdf_offsets, 
    float* sdf_shapes, float* weight, float* bias, int nkernels, int ncells, 
    float* kernel_size, float* dilation, float max_distance, float* out, float* dweight)
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
int cuda_convsdf(float* locs, int batch_size, int N, int ndims, float* idxs,
    float* poses, float* scales, int M, int pose_len, float* sdfs, float* sdf_offsets, 
    float* sdf_shapes, float* weight, float* bias, int nkernels, int ncells, 
    float* kernel_size, float* dilation, float max_distance, float* out, float* dweight, 
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
