
#ifdef __cplusplus
//extern "C" {
#endif

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "../external/cub-1.3.2/cub/cub.cuh"

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
        const float* qlocs,
		const float* locs, 
		const float* data, 
        const float* neighbors,
		const float* weight, 
		const float* bias, 
		const int batch_size, 
        const int M,
		const int N, 
		const int nchannels, 
		const int ndims, 
        const int max_neighbors,
		const int nkernels, 
		const int ncells, 
		const float radius, 
		const float* kernel_size, 
		const float* dilation, 
		const int dis_norm, 
		const int kernel_fn, 
		float* out, 
        float* dqlocs,
        float* dlocs,
		float* ddata, 
		float* dweight)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < M*batch_size; i += stride)
    {
    	int b = i/M;
    	int n = i%M;
    	compute_kernel_cells(qlocs, locs, data, neighbors, weight, bias, batch_size, M, N, 
    		nchannels, ndims, max_neighbors, nkernels, ncells, radius, kernel_size, dilation, 
    		dis_norm, kernel_fn, out, b, n, dqlocs, dlocs, ddata, dweight);
    }
}
int cuda_convsp(
        const float* qlocs,
        const float* locs, 
        const float* data, 
        const float* neighbors,
        const float* weight, 
        const float* bias, 
        const int batch_size, 
        const int M,
        const int N, 
        const int nchannels, 
        const int ndims, 
        const int max_neighbors,
        const int nkernels, 
        const int ncells, 
        const float radius, 
        const float* kernel_size, 
        const float* dilation, 
        const int dis_norm, 
        const int kernel_fn, 
        float* out, 
        float* dqlocs,
        float* dlocs,
        float* ddata, 
        float* dweight, 
        cudaStream_t stream, 
        const size_t nshared_device_mem)
{
	int nops = batch_size*M;
    int numBlocks = ceil(nops * (1.0/256));
    dim3 blocks(numBlocks);
    dim3 threads(256);

	kernel_convsp<<<blocks, threads, 0, stream>>>(qlocs, locs, data, neighbors, weight, bias,
		batch_size, M, N, nchannels, ndims, max_neighbors, nkernels, ncells, radius, 
        kernel_size, dilation, dis_norm, kernel_fn, out, dqlocs, dlocs, ddata, dweight);
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
    float* dlocs,
    float* dweight,
    float* dposes)
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
                    outk, dlocs, dweight, dposes, isdf_cache, fsdf_cache);
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
    float* dlocs,
    float* dweight, 
    float* dposes,
    cudaStream_t stream)
{
    int nops = batch_size*N*nkernels;
    int numBlocks = ceil(nops * (1.0/256));
    dim3 blocks(min(MAX_BLOCKS, numBlocks));
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
        kernel_size, dilation, max_distance, out, dlocs, dweight, dposes);
    cudaDeviceSynchronize();
    return PrintOnCudaError("cuda_convsdf");
}



// Functions for the ParticleCollision layer.
__global__
void kernel_compute_cellIDs(
    const float* locs,
    const float* low,
    const float* grid_dims,
    uint32_t* cellIDs,
    float* idxs,
    const int batch_size,
    const int N,
    const int ndims,
    const float cellEdge,
    uint32_t* maxhash)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, d;
    for(i = index; i < N*batch_size; i += stride)
    {
        int b = i/N;
        int n = i%N;
        int hash = 0;
        for(d = 0; d < ndims; ++d)
            hash += partial_grid_hash(
                loc2grid(locs[b*N*ndims + n*ndims + d], low[b*ndims + d], cellEdge), 
                grid_dims + b*ndims, d, ndims);
        cellIDs[i] = hash;
        idxs[i] = n;

        if(n == 0)
        {
            uint32_t mh = 0;
            for(d = 0; d < ndims; ++d)
                mh += partial_grid_hash(grid_dims[b*ndims + d] - 1, 
                    grid_dims + b*ndims, d, ndims);
            atomicMax(maxhash, mh);
        }
    }
}
int cuda_hashgrid_order(
    float* locs,
    const float* low,
    const float* grid_dims,
    float* cellIDs,
    float* idxs,
    float* buffer,
    const int batch_size,
    const int N,
    const int ndims,
    const float cellEdge,
    cudaStream_t stream)
{
    uint32_t* cellIDsi = (uint32_t*) cellIDs;
    int b;

    int nops = batch_size*N;
    int numBlocks = ceil(nops * (1.0/256));
    dim3 blocks(min(MAX_BLOCKS, numBlocks));
    dim3 threads(256); 
    kernel_compute_cellIDs<<<blocks, threads, 0, stream>>>(locs, low, grid_dims, cellIDsi,
        idxs, batch_size, N, ndims, cellEdge, (uint32_t*)buffer);
    cudaStreamSynchronize(stream);
    if(!PrintOnCudaError("cuda_hashgrid_order: kernel_compute_cellIDs")) return 0; 
    uint32_t maxhash;
    cudaMemcpy(&maxhash, buffer, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    uint32_t numBits = (uint32_t)ceil(log2((float)maxhash)) + 1;
    // TODO this seems to not work right, hard coding it at max val for now.
    numBits = sizeof(uint32_t)*8;

    // Sort the particles by cell ID.
    for(b = 0; b < batch_size; ++b)
    {
        cub::DoubleBuffer<uint32_t> d_keys(cellIDsi + b*N, cellIDsi + batch_size*N);
        cub::DoubleBuffer<float> d_values(idxs + b*N, cellIDs + (1 + batch_size)*N);

        size_t sortTempSize;
        cub::DeviceRadixSort::SortPairs(buffer, sortTempSize, d_keys, d_values, N, 0, 
            numBits, stream);
        cudaStreamSynchronize(stream);

        if (d_keys.Current() != cellIDsi + b*N)
            cudaMemcpyAsync(cellIDsi + b*N, d_keys.Current(), 
                sizeof(uint32_t)*N, cudaMemcpyDeviceToDevice, stream);

        if (d_values.Current() != idxs + b*N)
            cudaMemcpyAsync(idxs + b*N, d_values.Current(), sizeof(float)*N, 
                cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    // BUG: For some reason, CUDA won't finish the above cudaMemcpy's (async or 
    // otherwise) unless it copies some data to the heap (not the stack).
    float* didxs = new float;
    cudaMemcpy(didxs, idxs + b*N, sizeof(float), cudaMemcpyDeviceToHost);
    delete didxs;

    cudaDeviceSynchronize();
    return PrintOnCudaError("cuda_hashgrid_order");
}

__global__
void kernel_fill_cells(
    const uint32_t* cellIDs,
    float* cellStarts,
    float* cellEnds,
    const int batch_size,
    const int N,
    const int ncells)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    for(i = index; i < N*batch_size; i += stride)
    {
        int b = i/N;
        int n = i%N;
        int c = cellIDs[i];
        if(n == 0)
        {
            cellStarts[b*ncells + c] = n;
        }
        else
        {
            int p = cellIDs[b*N + n-1];

            if (c != p)
            {
                cellStarts[b*ncells + c] = n;
                cellEnds[b*ncells + p] = n;
            }
        }
        
        if(n == N-1)
        {
            cellEnds[b*ncells + c] = n+1;
        }
    }
}
__global__
void kernel_compute_collisions(
    const float* qlocs,
    const float* locs,
    const float* cellStarts,
    const float* cellEnds,
    const int batch_size,
    const int M,
    const int N,
    const int ndims,
    const int ncells,
    const float* low,
    const float* grid_dims,
    const float cellEdge,
    const float radius2,
    float* collisions,
    const int max_collisions,
    const int include_self)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    for(i = index; i < M*batch_size; i += stride)
    {
        int b = i/M;
        int n = i%M;
        compute_collisions(
            qlocs,
            locs,
            cellStarts,
            cellEnds,
            batch_size,
            M,
            N,
            ndims,
            ncells,
            low,
            grid_dims,
            cellEdge,
            radius2,
            collisions,
            max_collisions,
            include_self,
            b,
            n);
    }
}


int cuda_compute_collisions(
    const float* qlocs,
    const float* locs,
    const float* low,
    const float* grid_dims,
    const float* cellIDs,
    float* cellStarts,
    float* cellEnds,
    float* collisions,
    const int batch_size,
    const int M,
    const int N,
    const int ndims,
    const int max_collisions,
    const int ncells,
    const float cellEdge,
    const float radius,
    const int include_self,
    cudaStream_t stream)
{
    int nops = batch_size*N;
    int numBlocks = ceil(nops * (1.0/256));
    dim3 blocks(min(MAX_BLOCKS, numBlocks));
    dim3 threads(256); 

    const uint32_t* cellIDsi = (const uint32_t*) cellIDs;

    // Create the cell start and end lists.
    kernel_fill_cells<<<blocks, threads, 0, stream>>>(cellIDsi, cellStarts, cellEnds, 
        batch_size, N, ncells);
    cudaStreamSynchronize(stream);
    if(!PrintOnCudaError("compute_collisions")) return 0;

    nops = batch_size*N;
    numBlocks = ceil(nops * (1.0/256));
    blocks = dim3(min(MAX_BLOCKS, numBlocks));
    threads = dim3(256); 

    // Make collision lists.
    kernel_compute_collisions<<<blocks, threads, 0, stream>>>(
        qlocs,
        locs,
        cellStarts,
        cellEnds,
        batch_size,
        M,
        N,
        ndims,
        ncells,
        low,
        grid_dims,
        cellEdge,
        radius*radius,
        collisions,
        max_collisions,
        include_self);

    cudaStreamSynchronize(stream);
    return PrintOnCudaError("compute_collisions");
}

__global__
void kernel_reorder_data(
    const float* locs,
    const float* data,
    const float* idxs,
    float* nlocs,
    float* ndata,
    const int batch_size,
    const int N,
    const int ndims,
    const int nchannels,
    const int reverse)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, d;
    for(i = index; i < N*batch_size; i += stride)
    {
        int b = i/N;
        int nn = i%N;
        int on = idxs[i];
        if(reverse)
        {
            nn = idxs[i];
            on = i%N;
        }
        for(d = 0; d < ndims; ++d)
            nlocs[b*N*ndims + nn*ndims + d] = locs[b*N*ndims + on*ndims + d];
        for(d = 0; d < nchannels; ++d)
            ndata[b*N*nchannels + nn*nchannels + d] = data[b*N*nchannels + on*nchannels + d];
    }
}
int cuda_reorder_data(
    float* locs,
    float* data,
    float* idxs,
    float* nlocs,
    float* ndata,
    const int batch_size,
    const int N,
    const int ndims,
    const int nchannels,
    const int reverse,
    cudaStream_t stream)
{

    int nops = batch_size*N;
    int numBlocks = ceil(nops * (1.0/256));
    dim3 blocks(min(MAX_BLOCKS, numBlocks));
    dim3 threads(256); 
    

    // Re-order locs and data.
    kernel_reorder_data<<<blocks, threads, 0, stream>>>(locs, data, idxs, nlocs, 
        ndata, batch_size, N, ndims, nchannels, reverse);

    cudaDeviceSynchronize();
    return PrintOnCudaError("cuda_reorder_data");
}

size_t get_radixsort_buffer_size(cudaStream_t stream)
{
    cub::DoubleBuffer<int> d_keys(NULL, NULL);
    cub::DoubleBuffer<float> d_values(NULL, NULL);

    size_t sortTempSize;
    cub::DeviceRadixSort::SortPairs(NULL, sortTempSize, d_keys, d_values, 1, 0, 
        1, stream);
    return sortTempSize;
}

__global__
void kernel_particleprojection(
    const float* locs, 
    const float camera_fl,
    const float filter_std,
    const float filter_scale,
    const float* depth_mask, 
    const int batch_size,
    const int N,
    const int width,
    const int height,
    float* out, 
    float* dlocs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    for(i = index; i < N*batch_size; i += stride)
    {
        int b = i/N;
        int n = i%N;
        compute_particle_projection(
                locs,
                batch_size,
                N,
                camera_fl,
                width,
                height,
                filter_std,
                filter_scale,
                depth_mask,
                n,
                b,
                out,
                dlocs);
    }
}
int cuda_particleprojection(
        const float* locs, 
        const float camera_fl,
        const float filter_std,
        const float filter_scale,
        const float* depth_mask, 
        const int batch_size,
        const int N,
        const int width,
        const int height,
        float* out, 
        float* dlocs,
        cudaStream_t stream)
{
    int nops = batch_size*N;
    int numBlocks = ceil(nops * (1.0/256));
    dim3 blocks(min(MAX_BLOCKS, numBlocks));
    dim3 threads(256); 
    

    // Re-order locs and data.
    kernel_particleprojection<<<blocks, threads, 0, stream>>>(locs,
                                                              camera_fl,
                                                              filter_std,
                                                              filter_scale,
                                                              depth_mask,
                                                              batch_size,
                                                              N,
                                                              width,
                                                              height,
                                                              out,
                                                              dlocs);

    cudaDeviceSynchronize();
    return PrintOnCudaError("cuda_particleprojection");
}


__global__
void kernel_imageprojection(
    const float* locs, 
        const float* image,
        const float camera_fl,
        const float* depth_mask, 
        const int batch_size,
        const int N,
        const int width,
        const int height,
        const int channels,
        float* out, 
        float* dlocs,
        float* dimage)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    for(i = index; i < N*batch_size; i += stride)
    {
        int b = i/N;
        int n = i%N;
        compute_image_projection(
                locs,
                image,
                batch_size,
                N,
                camera_fl,
                width,
                height,
                channels,
                depth_mask,
                n,
                b,
                out,
                dlocs,
                dimage);
    }
}
int cuda_imageprojection(
        const float* locs, 
        const float* image,
        const float camera_fl,
        const float* depth_mask, 
        const int batch_size,
        const int N,
        const int width,
        const int height,
        const int channels,
        float* out, 
        float* dlocs,
        float* dimage,
        cudaStream_t stream)
{
    int nops = batch_size*N;
    int numBlocks = ceil(nops * (1.0/256));
    dim3 blocks(min(MAX_BLOCKS, numBlocks));
    dim3 threads(256); 
    

    // Re-order locs and data.
    kernel_imageprojection<<<blocks, threads, 0, stream>>>(locs,
                                                              image,
                                                              camera_fl,
                                                              depth_mask,
                                                              batch_size,
                                                              N,
                                                              width,
                                                              height,
                                                              channels,
                                                              out,
                                                              dlocs,
                                                              dimage);

    cudaDeviceSynchronize();
    return PrintOnCudaError("cuda_imageprojection");
}


#ifdef __cplusplus
//}
#endif

