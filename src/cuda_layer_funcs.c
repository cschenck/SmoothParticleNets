
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


int spnc_convsp_forward(const THCudaTensor* locs_t, const THCudaTensor* data_t, 
    const THCudaTensor* weight_t, const THCudaTensor* bias_t, const float radius, 
    const THCudaTensor* kernel_size_t, const THCudaTensor* dilation_t, const int dis_norm, 
    const int kernel_fn, THCudaTensor* out_t, const size_t nshared_device_mem)
{

    const float* locs = THCudaTensor_data(state, locs_t);
    const float* data = THCudaTensor_data(state, data_t);
    const float* weight = THCudaTensor_data(state, weight_t);
    const float* bias = THCudaTensor_data(state, bias_t);
    const int batch_size = locs_t->size[0];
    const int N = locs_t->size[1];
    const int nchannels = data_t->size[2];
    const int ndims = locs_t->size[2];
    const int nkernels = weight_t->size[0];
    const int ncells = weight_t->size[2];
    const float* kernel_size = THCudaTensor_data(state, kernel_size_t);
    const float* dilation = THCudaTensor_data(state, dilation_t);
    float* out = THCudaTensor_data(state, out_t);
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_convsp(locs, data, weight, bias, batch_size, N, nchannels, ndims,
        nkernels, ncells, radius, kernel_size, dilation, dis_norm, kernel_fn, out, NULL, 
        NULL, stream, nshared_device_mem);

}

int spnc_convsp_backward(const THCudaTensor* locs_t, const THCudaTensor* data_t, 
    const THCudaTensor* weight_t, const THCudaTensor* bias_t, const float radius, 
    const THCudaTensor* kernel_size_t, const THCudaTensor* dilation_t, const int dis_norm, 
    const int kernel_fn, THCudaTensor* out_t, THCudaTensor* ddata_t, THCudaTensor* dweight_t, 
    const size_t nshared_device_mem)
{
    const float* locs = THCudaTensor_data(state, locs_t);
    const float* data = THCudaTensor_data(state, data_t);
    const float* weight = THCudaTensor_data(state, weight_t);
    const float* bias = THCudaTensor_data(state, bias_t);
    float* ddata = THCudaTensor_data(state, ddata_t);
    float* dweight = THCudaTensor_data(state, dweight_t);
    const int batch_size = locs_t->size[0];
    const int N = locs_t->size[1];
    const int nchannels = data_t->size[2];
    const int ndims = locs_t->size[2];
    const int nkernels = weight_t->size[0];
    const int ncells = weight_t->size[2];
    const float* kernel_size = THCudaTensor_data(state, kernel_size_t);
    const float* dilation = THCudaTensor_data(state, dilation_t);
    float* out = THCudaTensor_data(state, out_t);
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_convsp(locs, data, weight, bias, batch_size, N, nchannels, ndims,
        nkernels, ncells, radius, kernel_size, dilation, dis_norm, kernel_fn, out, ddata, 
        dweight, stream, nshared_device_mem);
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

#else

int spnc_convsp_forward(void* locs_t, void* data_t, void* density_t, 
    void* weight_t, void* bias_t,float radius, void* kernel_size_t, 
    void* dilation_t, void* out_t)
{
    fprintf(stderr, "SmoothParticleNets was not compiled with Cuda suport.\n"
                     "Please recompile with the --with_cuda flag\n.");
    return 0;
}

int spnc_convsp_backward(void* locs_t, void* data_t, void* density_t, 
    void* weight_t, void* bias_t,float radius, void* kernel_size_t, 
    void* dilation_t, void* out_t, void* ddata_t,
    void* dweight_t)
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

int spnc_get_shared_mem_size(int device)
{
    fprintf(stderr, "SmoothParticleNets was not compiled with Cuda suport.\n"
                     "Please recompile with the --with_cuda flag\n.");
    return 0;
}

#endif