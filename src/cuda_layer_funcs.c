
#include <stdio.h>

#include <TH/TH.h>


#ifdef WITH_CUDA
#include <THC/THC.h>

#include "gpu_kernels.h"

extern THCState *state;



int spnc_convsp_forward(THCudaTensor* locs_t, THCudaTensor* data_t, THCudaTensor* density_t, 
    THCudaTensor* neighborlist_t, THCudaTensor* weight_t, THCudaTensor* bias_t, float radius, 
    THCudaTensor* kernel_size_t, THCudaTensor* dilation_t, THCudaTensor* out_t)
{

    float* locs = THCudaTensor_data(state, locs_t);
    float* data = THCudaTensor_data(state, data_t);
    float* density = THCudaTensor_data(state, density_t);
    float* neighborlist = THCudaTensor_data(state, neighborlist_t);
    float* weight = THCudaTensor_data(state, weight_t);
    float* bias = THCudaTensor_data(state, bias_t);
    int batch_size = locs_t->size[0];
    int N = locs_t->size[1];
    int nchannels = data_t->size[1];
    int ndims = locs_t->size[2] - 1;
    int nkernels = weight_t->size[0];
    int ncells = weight_t->size[2];
    int nneighbors = neighborlist_t->size[2];
    float* kernel_size = THCudaTensor_data(state, kernel_size_t);
    float* dilation = THCudaTensor_data(state, dilation_t);
    float* out = THCudaTensor_data(state, out_t);
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_convsp(locs, data, density, neighborlist, weight, bias, batch_size, N, 
        nchannels, ndims, nneighbors,
        nkernels, ncells, radius, kernel_size, dilation, out, NULL, NULL, stream);

}

int spnc_convsp_backward(THCudaTensor* locs_t, THCudaTensor* data_t, THCudaTensor* density_t, 
    THCudaTensor* neighborlist_t, THCudaTensor* weight_t, THCudaTensor* bias_t, float radius, 
    THCudaTensor* kernel_size_t, THCudaTensor* dilation_t, THCudaTensor* out_t,
    THCudaTensor* ddata_t, THCudaTensor* dweight_t)
{
    float* locs = THCudaTensor_data(state, locs_t);
    float* data = THCudaTensor_data(state, data_t);
    float* density = THCudaTensor_data(state, density_t);
    float* neighborlist = THCudaTensor_data(state, neighborlist_t);
    float* weight = THCudaTensor_data(state, weight_t);
    float* bias = THCudaTensor_data(state, bias_t);
    float* ddata = THCudaTensor_data(state, ddata_t);
    float* dweight = THCudaTensor_data(state, dweight_t);
    int batch_size = locs_t->size[0];
    int N = locs_t->size[1];
    int nchannels = data_t->size[1];
    int ndims = locs_t->size[2] - 1;
    int nkernels = weight_t->size[0];
    int ncells = weight_t->size[2];
    int nneighbors = neighborlist_t->size[2];
    float* kernel_size = THCudaTensor_data(state, kernel_size_t);
    float* dilation = THCudaTensor_data(state, dilation_t);
    float* out = THCudaTensor_data(state, out_t);
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_convsp(locs, data, density, neighborlist, weight, bias, batch_size, N, 
        nchannels, ndims, nneighbors,
        nkernels, ncells, radius, kernel_size, dilation, out, ddata, dweight, stream);
}

int spnc_convsdf_forward(THCudaTensor* locs_t, THCudaTensor* idxs_t, THCudaTensor* poses_t, 
    THCudaTensor* scales_t, THCudaTensor* sdfs_t, THCudaTensor* sdf_offsets_t,
    THCudaTensor* sdf_shapes_t, THCudaTensor* weight_t, THCudaTensor* bias_t, 
    THCudaTensor* kernel_size_t, THCudaTensor* dilation_t, float max_distance,
    THCudaTensor* out_t)
{
    float* locs = THCudaTensor_data(state, locs_t);
    float* idxs = THCudaTensor_data(state, idxs_t);
    float* poses = THCudaTensor_data(state, poses_t);
    float* scales = THCudaTensor_data(state, scales_t);
    float* sdfs = THCudaTensor_data(state, sdfs_t);
    float* sdf_offsets = THCudaTensor_data(state, sdf_offsets_t);
    float* sdf_shapes = THCudaTensor_data(state, sdf_shapes_t);
    float* weight = THCudaTensor_data(state, weight_t);
    float* bias = THCudaTensor_data(state, bias_t); 
    int batch_size = locs_t->size[0];
    int N = locs_t->size[1];
    int ndims = locs_t->size[2] - 1;
    int M = idxs_t->size[1];
    int pose_len = poses_t->size[2];
    int nkernels = weight_t->size[0];
    int ncells = weight_t->size[1];
    float* kernel_size = THCudaTensor_data(state, kernel_size_t);
    float* dilation = THCudaTensor_data(state, dilation_t);
    float* out = THCudaTensor_data(state, out_t);
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_convsdf(locs, batch_size, N, ndims, idxs, poses, scales, M, pose_len, sdfs, 
        sdf_offsets, sdf_shapes, weight, bias, nkernels, ncells, kernel_size, dilation, 
        max_distance, out, NULL, stream);
}

int spnc_convsdf_backward(THCudaTensor* locs_t, THCudaTensor* idxs_t, THCudaTensor* poses_t, 
    THCudaTensor* scales_t, THCudaTensor* sdfs_t, THCudaTensor* sdf_offsets_t,
    THCudaTensor* sdf_shapes_t, THCudaTensor* weight_t, THCudaTensor* bias_t, 
    THCudaTensor* kernel_size_t, THCudaTensor* dilation_t, float max_distance,
    THCudaTensor* out_t, THCudaTensor* dweight_t)
{
    float* locs = THCudaTensor_data(state, locs_t);
    float* idxs = THCudaTensor_data(state, idxs_t);
    float* poses = THCudaTensor_data(state, poses_t);
    float* scales = THCudaTensor_data(state, scales_t);
    float* sdfs = THCudaTensor_data(state, sdfs_t);
    float* sdf_offsets = THCudaTensor_data(state, sdf_offsets_t);
    float* sdf_shapes = THCudaTensor_data(state, sdf_shapes_t);
    float* weight = THCudaTensor_data(state, weight_t);
    float* bias = THCudaTensor_data(state, bias_t); 
    float* dweight = THCudaTensor_data(state, dweight_t); 
    int batch_size = locs_t->size[0];
    int N = locs_t->size[1];
    int ndims = locs_t->size[2] - 1;
    int M = idxs_t->size[1];
    int pose_len = poses_t->size[2];
    int nkernels = weight_t->size[0];
    int ncells = weight_t->size[1];
    float* kernel_size = THCudaTensor_data(state, kernel_size_t);
    float* dilation = THCudaTensor_data(state, dilation_t);
    float* out = THCudaTensor_data(state, out_t);
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_convsdf(locs, batch_size, N, ndims, idxs, poses, scales, M, pose_len, sdfs, 
        sdf_offsets, sdf_shapes, weight, bias, nkernels, ncells, kernel_size, dilation, 
        max_distance, out, dweight, stream);
}

int spnc_neighborlist(THCudaTensor* locs_t, float radius, THCudaTensor* neighborlist_t)
{
    float* locs = THCudaTensor_data(state, locs_t);
    float* neighborlist = THCudaTensor_data(state, neighborlist_t);
    int batch_size = locs_t->size[0];
    int N = locs_t->size[1];
    int ndims = locs_t->size[2] - 1;
    int nneighbors = neighborlist_t->size[2];
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_neighborlist(locs, neighborlist, batch_size, N, ndims, nneighbors, 
        radius, stream);
}

#else

int spnc_convsp_forward(void* locs_t, void* data_t, void* density_t, 
    void* neighborlist_t, void* weight_t, void* bias_t, float radius, 
    void* kernel_size_t, void* dilation_t, void* out_t)
{
    fprintf(stderr, "SmoothParticleNets was not compiled with Cuda suport.\n"
                     "Please recompile with the --with_cuda flag\n.");
    return 0;
}

int spnc_convsp_backward(void* locs_t, void* data_t, void* density_t, 
    void* neighborlist_t, void* weight_t, void* bias_t, float radius, 
    void* kernel_size_t, void* dilation_t, void* out_t,
    void* ddata_t, void* dweight_t)
{
    fprintf(stderr, "SmoothParticleNets was not compiled with Cuda suport.\n"
                     "Please recompile with the --with_cuda flag\n.");
    return 0;
}

int spnc_convsdf_forward(void** locs_t, void** idxs_t, void** poses_t, 
    void** scales_t, void** sdfs_t, void** sdf_offsets_t,
    void** sdf_shapes_t, void** weight_t, void** bias_t, 
    void** kernel_size_t, void** dilation_t, float max_distance,
    void** out_t)
{
    fprintf(stderr, "SmoothParticleNets was not compiled with Cuda suport.\n"
                     "Please recompile with the --with_cuda flag\n.");
    return 0;
}

int spnc_convsdf_backward(void** locs_t, void** idxs_t, void** poses_t, 
    void** scales_t, void** sdfs_t, void** sdf_offsets_t,
    void** sdf_shapes_t, void** weight_t, void** bias_t, 
    void** kernel_size_t, void** dilation_t, float max_distance,
    void** out_t, void** dweight_t)
{
    fprintf(stderr, "SmoothParticleNets was not compiled with Cuda suport.\n"
                     "Please recompile with the --with_cuda flag\n.");
    return 0;
}

int spnc_neighborlist(void* locs_t, float radius, void* neighborlist_t)
{
    fprintf(stderr, "SmoothParticleNets was not compiled with Cuda suport.\n"
                     "Please recompile with the --with_cuda flag\n.");
    return 0;
}

#endif