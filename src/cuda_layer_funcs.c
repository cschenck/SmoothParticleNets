
#include <stdio.h>

#include <TH/TH.h>


#ifdef WITH_CUDA
#include <THC/THC.h>

#include "gpu_kernels.h"

extern THCState *state;


size_t spnc_get_shared_mem_size(int device)
{
    return GetSharedMemPerBlock(device);
}


int spnc_convsp_forward(const THCudaTensor* qlocs_t, const THCudaTensor* locs_t, 
    const THCudaTensor* data_t, const THCudaTensor* neighbors_t,
    const THCudaTensor* weight_t, const THCudaTensor* bias_t, const float radius, 
    const THCudaTensor* kernel_size_t, const THCudaTensor* dilation_t, const int dis_norm, 
    const int kernel_fn, THCudaTensor* out_t, const size_t nshared_device_mem)
{
    const float* qlocs = THCudaTensor_data(state, qlocs_t);
    const float* locs = THCudaTensor_data(state, locs_t);
    const float* data = THCudaTensor_data(state, data_t);
    const float* neighbors = THCudaTensor_data(state, neighbors_t);
    const float* weight = THCudaTensor_data(state, weight_t);
    const float* bias = THCudaTensor_data(state, bias_t);
    const int batch_size = locs_t->size[0];
    const int M = qlocs_t->size[1];
    const int N = locs_t->size[1];
    const int nchannels = data_t->size[2];
    const int ndims = locs_t->size[2];
    const int max_neighbors = neighbors_t->size[2];
    const int nkernels = weight_t->size[0];
    const int ncells = weight_t->size[2];
    const float* kernel_size = THCudaTensor_data(state, kernel_size_t);
    const float* dilation = THCudaTensor_data(state, dilation_t);
    float* out = THCudaTensor_data(state, out_t);
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_convsp(qlocs, locs, data, neighbors, weight, bias, batch_size, M, N, nchannels, ndims,
        max_neighbors, nkernels, ncells, radius, kernel_size, dilation, dis_norm, kernel_fn, 
        out, NULL, NULL, NULL, NULL, stream, nshared_device_mem);

}

int spnc_convsp_backward(const THCudaTensor* qlocs_t, const THCudaTensor* locs_t, 
    const THCudaTensor* data_t, const THCudaTensor* neighbors_t,
    const THCudaTensor* weight_t, const THCudaTensor* bias_t, const float radius, 
    const THCudaTensor* kernel_size_t, const THCudaTensor* dilation_t, const int dis_norm, 
    const int kernel_fn, THCudaTensor* out_t, THCudaTensor* dqlocs_t, THCudaTensor* dlocs_t, 
    THCudaTensor* ddata_t, THCudaTensor* dweight_t, const size_t nshared_device_mem)
{
    const float* qlocs = THCudaTensor_data(state, qlocs_t);
    const float* locs = THCudaTensor_data(state, locs_t);
    const float* data = THCudaTensor_data(state, data_t);
    const float* neighbors = THCudaTensor_data(state, neighbors_t);
    const float* weight = THCudaTensor_data(state, weight_t);
    const float* bias = THCudaTensor_data(state, bias_t);
    float* dqlocs = THCudaTensor_data(state, dqlocs_t);
    float* dlocs = THCudaTensor_data(state, dlocs_t);
    float* ddata = THCudaTensor_data(state, ddata_t);
    float* dweight = THCudaTensor_data(state, dweight_t);
    const int batch_size = locs_t->size[0];
    const int M = qlocs_t->size[1];
    const int N = locs_t->size[1];
    const int nchannels = data_t->size[2];
    const int ndims = locs_t->size[2];
    const int max_neighbors = neighbors_t->size[2];
    const int nkernels = weight_t->size[0];
    const int ncells = weight_t->size[2];
    const float* kernel_size = THCudaTensor_data(state, kernel_size_t);
    const float* dilation = THCudaTensor_data(state, dilation_t);
    float* out = THCudaTensor_data(state, out_t);
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_convsp(qlocs, locs, data, neighbors, weight, bias, batch_size, M, N, nchannels, ndims,
        max_neighbors, nkernels, ncells, radius, kernel_size, dilation, dis_norm, kernel_fn, 
        out, dqlocs, dlocs, ddata, dweight, stream, nshared_device_mem);
}

int spnc_convsdf_forward(const THCudaTensor* locs_t, const THCudaTensor* idxs_t, 
    const THCudaTensor* poses_t, const THCudaTensor* scales_t, const THCudaTensor* sdfs_t, 
    const THCudaTensor* sdf_offsets_t, const THCudaTensor* sdf_shapes_t, 
    const THCudaTensor* weight_t, const THCudaTensor* bias_t, 
    const THCudaTensor* kernel_size_t, const THCudaTensor* dilation_t, 
    const float max_distance, THCudaTensor* out_t)
{
    const float* locs = THCudaTensor_data(state, locs_t);
    const float* idxs = THCudaTensor_data(state, idxs_t);
    const float* poses = THCudaTensor_data(state, poses_t);
    const float* scales = THCudaTensor_data(state, scales_t);
    const float* sdfs = THCudaTensor_data(state, sdfs_t);
    const float* sdf_offsets = THCudaTensor_data(state, sdf_offsets_t);
    const float* sdf_shapes = THCudaTensor_data(state, sdf_shapes_t);
    const float* weight = THCudaTensor_data(state, weight_t);
    const float* bias = THCudaTensor_data(state, bias_t); 
    const int batch_size = locs_t->size[0];
    const int N = locs_t->size[1];
    const int ndims = locs_t->size[2];
    const int M = idxs_t->size[1];
    const int pose_len = poses_t->size[2];
    const int nkernels = weight_t->size[0];
    const int ncells = weight_t->size[1];
    const float* kernel_size = THCudaTensor_data(state, kernel_size_t);
    const float* dilation = THCudaTensor_data(state, dilation_t);
    float* out = THCudaTensor_data(state, out_t);
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_convsdf(locs, batch_size, N, ndims, idxs, poses, scales, M, pose_len, sdfs, 
        sdf_offsets, sdf_shapes, weight, bias, nkernels, ncells, kernel_size, dilation, 
        max_distance, out, NULL, stream);
}

int spnc_convsdf_backward(const THCudaTensor* locs_t, const THCudaTensor* idxs_t, 
    const THCudaTensor* poses_t, const THCudaTensor* scales_t, const THCudaTensor* sdfs_t, 
    const THCudaTensor* sdf_offsets_t, const THCudaTensor* sdf_shapes_t, 
    const THCudaTensor* weight_t, const THCudaTensor* bias_t, 
    const THCudaTensor* kernel_size_t, const THCudaTensor* dilation_t, 
    const float max_distance, THCudaTensor* out_t, THCudaTensor* dweight_t)
{
    const float* locs = THCudaTensor_data(state, locs_t);
    const float* idxs = THCudaTensor_data(state, idxs_t);
    const float* poses = THCudaTensor_data(state, poses_t);
    const float* scales = THCudaTensor_data(state, scales_t);
    const float* sdfs = THCudaTensor_data(state, sdfs_t);
    const float* sdf_offsets = THCudaTensor_data(state, sdf_offsets_t);
    const float* sdf_shapes = THCudaTensor_data(state, sdf_shapes_t);
    const float* weight = THCudaTensor_data(state, weight_t);
    const float* bias = THCudaTensor_data(state, bias_t); 
    float* dweight = THCudaTensor_data(state, dweight_t); 
    const int batch_size = locs_t->size[0];
    const int N = locs_t->size[1];
    const int ndims = locs_t->size[2];
    const int M = idxs_t->size[1];
    const int pose_len = poses_t->size[2];
    const int nkernels = weight_t->size[0];
    const int ncells = weight_t->size[1];
    const float* kernel_size = THCudaTensor_data(state, kernel_size_t);
    const float* dilation = THCudaTensor_data(state, dilation_t);
    float* out = THCudaTensor_data(state, out_t);
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_convsdf(locs, batch_size, N, ndims, idxs, poses, scales, M, pose_len, sdfs, 
        sdf_offsets, sdf_shapes, weight, bias, nkernels, ncells, kernel_size, dilation, 
        max_distance, out, dweight, stream);
}

int spnc_hashgrid_order(THCudaTensor* locs_t, 
                           THCudaTensor* lower_bounds_t,
                           THCudaTensor* grid_dims_t,
                           THCudaTensor* cellIDs_t,
                           THCudaTensor* idxs_t,
                           THCudaTensor* buffer_t,
                           const float cellEdge)
{
    float* locs = THCudaTensor_data(state, locs_t);
    float* low = THCudaTensor_data(state, lower_bounds_t);
    float* grid_dims = THCudaTensor_data(state, grid_dims_t);
    float* cellIDs = THCudaTensor_data(state, cellIDs_t);
    float* idxs = THCudaTensor_data(state, idxs_t);
    float* buffer = THCudaTensor_data(state, buffer_t);
    const int batch_size = locs_t->size[0];
    const int N = locs_t->size[1];
    const int ndims = locs_t->size[2];
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_hashgrid_order(locs, low, grid_dims, cellIDs, idxs,
        buffer, batch_size, N, ndims, cellEdge, stream);
}

int spnc_compute_collisions(THCudaTensor* qlocs_t, 
                           THCudaTensor* locs_t, 
                           THCudaTensor* lower_bounds_t,
                           THCudaTensor* grid_dims_t,
                           THCudaTensor* cellIDs_t,
                           THCudaTensor* cellStarts_t,
                           THCudaTensor* cellEnds_t,
                           THCudaTensor* collisions_t,
                           const float cellEdge,
                           const float radius,
                           const int include_self)
{
    float* qlocs = THCudaTensor_data(state, qlocs_t);
    float* locs = THCudaTensor_data(state, locs_t);
    float* low = THCudaTensor_data(state, lower_bounds_t);
    float* grid_dims = THCudaTensor_data(state, grid_dims_t);
    float* cellIDs = THCudaTensor_data(state, cellIDs_t);
    float* cellStarts = THCudaTensor_data(state, cellStarts_t);
    float* cellEnds = THCudaTensor_data(state, cellEnds_t);
    float* collisions = THCudaTensor_data(state, collisions_t);
    const int batch_size = locs_t->size[0];
    const int M = qlocs_t->size[1];
    const int N = locs_t->size[1];
    const int ndims = locs_t->size[2];
    const int max_collisions = collisions_t->size[2];
    const int ncells = cellStarts_t->size[1];
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_compute_collisions(qlocs, locs, low, grid_dims, cellIDs, cellStarts,
        cellEnds, collisions, batch_size, M, N, ndims, max_collisions, 
        ncells, cellEdge, radius, include_self, stream);
}

int spnc_reorder_data(THCudaTensor* locs_t, 
                         THCudaTensor* data_t, 
                         THCudaTensor* idxs_t,
                         THCudaTensor* nlocs_t,
                         THCudaTensor* ndata_t,
                         const int reverse)
{
    float* locs = THCudaTensor_data(state, locs_t);
    float* data = THCudaTensor_data(state, data_t);
    float* idxs = THCudaTensor_data(state, idxs_t);
    float* nlocs = THCudaTensor_data(state, nlocs_t);
    float* ndata = THCudaTensor_data(state, ndata_t);
    const int batch_size = locs_t->size[0];
    const int N = locs_t->size[1];
    const int ndims = locs_t->size[2];
    int nchannels = 0;
    if(data_t->nDimension > 0)
        nchannels = data_t->size[2];
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_reorder_data(
            locs,
            data,
            idxs,
            nlocs,
            ndata,
            batch_size,
            N,
            ndims,
            nchannels,
            reverse,
            stream);
}

size_t spnc_get_radixsort_buffer_size(void)
{
    cudaStream_t stream = THCState_getCurrentStream(state);
    return get_radixsort_buffer_size(stream);
}


#else

int spnc_convsp_forward(const void* qlocs_t, const void* locs_t, const void* data_t, 
    const void* neighbors_t,
    const void* weight_t, const void* bias_t, const float radius, 
    const void* kernel_size_t, const void* dilation_t, const int dis_norm, 
    const int kernel_fn, void* out_t, const size_t nshared_device_mem)
{
    fprintf(stderr, "SmoothParticleNets was not compiled with Cuda suport.\n"
                     "Please recompile with the --with_cuda flag\n.");
    return 0;
}

int spnc_convsp_backward(const void* qlocs_t, const void* locs_t, const void* data_t, 
    const void* neighbors_t,
    const void* weight_t, const void* bias_t, const float radius, 
    const void* kernel_size_t, const void* dilation_t, const int dis_norm, 
    const int kernel_fn, void* out_t, void* dqlocs_t, void* dlocs_t, 
    void* ddata_t, void* dweight_t, const size_t nshared_device_mem)
{
    fprintf(stderr, "SmoothParticleNets was not compiled with Cuda suport.\n"
                     "Please recompile with the --with_cuda flag\n.");
    return 0;
}

int spnc_convsdf_forward(const void* locs_t, const void* idxs_t, 
    const void* poses_t, const void* scales_t, const void* sdfs_t, 
    const void* sdf_offsets_t, const void* sdf_shapes_t, 
    const void* weight_t, const void* bias_t, 
    const void* kernel_size_t, const void* dilation_t, 
    const float max_distance, void* out_t)
{
    fprintf(stderr, "SmoothParticleNets was not compiled with Cuda suport.\n"
                     "Please recompile with the --with_cuda flag\n.");
    return 0;
}

int spnc_convsdf_backward(const void* locs_t, const void* idxs_t, 
    const void* poses_t, const void* scales_t, const void* sdfs_t, 
    const void* sdf_offsets_t, const void* sdf_shapes_t, 
    const void* weight_t, const void* bias_t, 
    const void* kernel_size_t, const void* dilation_t, 
    const float max_distance, void* out_t, void* dweight_t)
{
    fprintf(stderr, "SmoothParticleNets was not compiled with Cuda suport.\n"
                     "Please recompile with the --with_cuda flag\n.");
    return 0;
}

int spnc_get_shared_mem_size(int device)
{
    fprintf(stderr, "SmoothParticleNets was not compiled with Cuda suport.\n"
                     "Please recompile with the --with_cuda flag\n.");
    return 0;
}

int spnc_hashgrid_order(void* locs_t, 
                           void* lower_bounds_t,
                           void* grid_dims_t,
                           void* cellIDs_t,
                           void* idxs_t,
                           void* buffer_t,
                           const float cellEdge)
{        
    fprintf(stderr, "SmoothParticleNets was not compiled with Cuda suport.\n"
                     "Please recompile with the --with_cuda flag\n.");
    return 0;
}

int spnc_compute_collisions(void* qlocs_t,
                           void* locs_t, 
                           void* lower_bounds_t,
                           void* grid_dims_t,
                           void* cellIDs_t,
                           void* cellStarts_t,
                           void* cellEnds_t,
                           void* collisions_t,
                           const float cellEdge,
                           const float radius,
                           const int include_self)
{
    fprintf(stderr, "SmoothParticleNets was not compiled with Cuda suport.\n"
                     "Please recompile with the --with_cuda flag\n.");
    return 0;
}

int spnc_reorder_data(void* locs_t, 
                         void* data_t, 
                         void* idxs_t,
                         void* buffer_t,
                         const int reverse)
{
    fprintf(stderr, "SmoothParticleNets was not compiled with Cuda suport.\n"
                     "Please recompile with the --with_cuda flag\n.");
    return 0;
}

size_t spnc_get_radixsort_buffer_size(void)
{
    fprintf(stderr, "SmoothParticleNets was not compiled with Cuda suport.\n"
                     "Please recompile with the --with_cuda flag\n.");
    return 0;
}

#endif