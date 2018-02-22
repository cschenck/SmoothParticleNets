#include <TH/TH.h>

#ifdef CUDA
#undef CUDA
#endif
#include "common_funcs.h"
#include "constants.h"

int cpu_convsp(float* locs, float* data, float* density, float* weight, float* bias, 
    int batch_size, int N, int nchannels, int ndims, int nkernels, int ncells, 
    float radius, float* kernel_size, float* dilation, float* out, float* ddata, 
    float* dweight);

int cpu_convsdf(float* locs, int batch_size, int N, int ndims, float* idxs,
    float* poses, float* scales, int M, int pose_len, float* sdfs, float* sdf_offsets, 
    float* sdf_shapes, float* weight, float* bias, int nkernels, int ncells, 
    float* kernel_size, float* dilation, float max_distance, float* out, float* dweight);


int spn_max_cartesian_dim(void)
{
    return MAX_CARTESIAN_DIM;
}

int spn_convsp_forward(THFloatTensor* locs_t, THFloatTensor* data_t, THFloatTensor* density_t, 
    THFloatTensor* weight_t, THFloatTensor* bias_t,float radius, THFloatTensor* kernel_size_t, 
    THFloatTensor* dilation_t, THFloatTensor* out_t)
{

    float* locs = THFloatTensor_data(locs_t);
    float* data = THFloatTensor_data(data_t);
    float* density = THFloatTensor_data(density_t);
    float* weight = THFloatTensor_data(weight_t);
    float* bias = THFloatTensor_data(bias_t); 
    int batch_size = locs_t->size[0];
    int N = locs_t->size[1];
    int nchannels = data_t->size[2];
    int ndims = locs_t->size[2] - 1;
    int nkernels = weight_t->size[0];
    int ncells = weight_t->size[2];
    float* kernel_size = THFloatTensor_data(kernel_size_t);
    float* dilation = THFloatTensor_data(dilation_t);
    float* out = THFloatTensor_data(out_t);

    return cpu_convsp(locs, data, density, weight, bias, batch_size, N, nchannels, ndims,
        nkernels, ncells, radius, kernel_size, dilation, out, NULL, NULL);
}

int spn_convsp_backward(THFloatTensor* locs_t, THFloatTensor* data_t, THFloatTensor* density_t, 
    THFloatTensor* weight_t, THFloatTensor* bias_t,float radius, THFloatTensor* kernel_size_t, 
    THFloatTensor* dilation_t, THFloatTensor* out_t, THFloatTensor* ddata_t,
    THFloatTensor* dweight_t)
{

    float* locs = THFloatTensor_data(locs_t);
    float* data = THFloatTensor_data(data_t);
    float* density = THFloatTensor_data(density_t);
    float* weight = THFloatTensor_data(weight_t);
    float* bias = THFloatTensor_data(bias_t); 
    float* ddata = THFloatTensor_data(ddata_t);
    float* dweight = THFloatTensor_data(dweight_t);
    int batch_size = locs_t->size[0];
    int N = locs_t->size[1];
    int nchannels = data_t->size[2];
    int ndims = locs_t->size[2] - 1;
    int nkernels = weight_t->size[0];
    int ncells = weight_t->size[2];
    float* kernel_size = THFloatTensor_data(kernel_size_t);
    float* dilation = THFloatTensor_data(dilation_t);
    float* out = THFloatTensor_data(out_t);

    return cpu_convsp(locs, data, density, weight, bias, batch_size, N, nchannels, ndims,
        nkernels, ncells, radius, kernel_size, dilation, out, ddata, dweight);
}

int cpu_convsp(float* locs, float* data, float* density, float* weight, float* bias, 
    int batch_size, int N, int nchannels, int ndims, int nkernels, int ncells, 
    float radius, float* kernel_size, float* dilation, float* out, float* ddata, 
    float* dweight)
{
    int b, n;
    for(b = 0; b < batch_size; ++b)
    {
        for(n = 0; n < N; ++n)
        {
            compute_kernel_cells(locs, data, density, weight, bias, batch_size, N, 
                nchannels, ndims, nkernels, ncells, radius, kernel_size, dilation, 
                out, b, n, 0, N, ddata, dweight);
        }
    }
    return 1;
}




int spn_convsdf_forward(THFloatTensor* locs_t, THFloatTensor* idxs_t, THFloatTensor* poses_t, 
    THFloatTensor* scales_t, THFloatTensor* sdfs_t, THFloatTensor* sdf_offsets_t,
    THFloatTensor* sdf_shapes_t, THFloatTensor* weight_t, THFloatTensor* bias_t, 
    THFloatTensor* kernel_size_t, THFloatTensor* dilation_t, float max_distance,
    THFloatTensor* out_t)
{

    float* locs = THFloatTensor_data(locs_t);
    float* idxs = THFloatTensor_data(idxs_t);
    float* poses = THFloatTensor_data(poses_t);
    float* scales = THFloatTensor_data(scales_t);
    float* sdfs = THFloatTensor_data(sdfs_t);
    float* sdf_offsets = THFloatTensor_data(sdf_offsets_t);
    float* sdf_shapes = THFloatTensor_data(sdf_shapes_t);
    float* weight = THFloatTensor_data(weight_t);
    float* bias = THFloatTensor_data(bias_t); 
    int batch_size = locs_t->size[0];
    int N = locs_t->size[1];
    int ndims = locs_t->size[2] - 1;
    int M = idxs_t->size[1];
    int pose_len = poses_t->size[2];
    int nkernels = weight_t->size[0];
    int ncells = weight_t->size[1];
    float* kernel_size = THFloatTensor_data(kernel_size_t);
    float* dilation = THFloatTensor_data(dilation_t);
    float* out = THFloatTensor_data(out_t);

    return cpu_convsdf(locs, batch_size, N, ndims, idxs, poses, scales, M, pose_len, sdfs, 
        sdf_offsets, sdf_shapes, weight, bias, nkernels, ncells, kernel_size, dilation, 
        max_distance, out, NULL);
}

int spn_convsdf_backward(THFloatTensor* locs_t, THFloatTensor* idxs_t, THFloatTensor* poses_t, 
    THFloatTensor* scales_t, THFloatTensor* sdfs_t, THFloatTensor* sdf_offsets_t,
    THFloatTensor* sdf_shapes_t, THFloatTensor* weight_t, THFloatTensor* bias_t, 
    THFloatTensor* kernel_size_t, THFloatTensor* dilation_t, float max_distance,
    THFloatTensor* out_t, THFloatTensor* dweight_t)
{

    float* locs = THFloatTensor_data(locs_t);
    float* idxs = THFloatTensor_data(idxs_t);
    float* poses = THFloatTensor_data(poses_t);
    float* scales = THFloatTensor_data(scales_t);
    float* sdfs = THFloatTensor_data(sdfs_t);
    float* sdf_offsets = THFloatTensor_data(sdf_offsets_t);
    float* sdf_shapes = THFloatTensor_data(sdf_shapes_t);
    float* weight = THFloatTensor_data(weight_t);
    float* bias = THFloatTensor_data(bias_t); 
    float* dweight = THFloatTensor_data(dweight_t); 
    int batch_size = locs_t->size[0];
    int N = locs_t->size[1];
    int ndims = locs_t->size[2] - 1;
    int M = idxs_t->size[1];
    int pose_len = poses_t->size[2];
    int nkernels = weight_t->size[0];
    int ncells = weight_t->size[1];
    float* kernel_size = THFloatTensor_data(kernel_size_t);
    float* dilation = THFloatTensor_data(dilation_t);
    float* out = THFloatTensor_data(out_t);

    return cpu_convsdf(locs, batch_size, N, ndims, idxs, poses, scales, M, pose_len, sdfs, 
        sdf_offsets, sdf_shapes, weight, bias, nkernels, ncells, kernel_size, dilation, 
        max_distance, out, dweight);
}

int cpu_convsdf(float* locs, int batch_size, int N, int ndims, float* idxs,
    float* poses, float* scales, int M, int pose_len, float* sdfs, float* sdf_offsets, 
    float* sdf_shapes, float* weight, float* bias, int nkernels, int ncells, 
    float* kernel_size, float* dilation, float max_distance, float* out, float* dweight)
{
    int* isdf_cache = (int*)malloc(sizeof(int)*M);
    float* fsdf_cache = (float*)malloc(sizeof(float)*M);
    int b, n, outk;
    for(b = 0; b < batch_size; ++b)
    {
        for(n = 0; n < N; ++n)
        {
            for(outk = 0; outk < nkernels; ++outk)
            {
                compute_sdf_kernel_cells(locs, batch_size, N, ndims, idxs, poses, 
                    scales, M, pose_len, sdfs, sdf_offsets, sdf_shapes, weight, bias, 
                    nkernels, ncells, kernel_size, dilation, max_distance, out, b, n,
                    outk, dweight, isdf_cache, fsdf_cache);
            }
        }
    }
    free(isdf_cache);
    free(fsdf_cache);
    return 1;
}