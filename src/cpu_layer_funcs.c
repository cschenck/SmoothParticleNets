#include <TH/TH.h>

#ifdef CUDA
#undef CUDA
#endif
#include "common_funcs.h"
#include "constants.h"


int cpu_convsp(const float* locs, const float* data, const float* weight, const float* bias, 
    const int batch_size, const int N, const int nchannels, const int ndims, 
    const int nkernels, const int ncells, const float radius, const float* kernel_size, 
    const float* dilation, const int dis_norm, const int kernel_fn, float* out, 
    float* ddata, float* dweight);

int cpu_convsdf(const float* locs, const int batch_size, const int N, const int ndims, 
    const float* idxs, const float* poses, const float* scales, const int M, 
    const int pose_len, const float* sdfs, const float* sdf_offsets, const float* sdf_shapes, 
    const float* weight, const float* bias, const int nkernels, const int ncells, 
    const float* kernel_size, const float* dilation, const float max_distance, 
    float* out, float* dweight);


int spn_max_cartesian_dim(void)
{
    return MAX_CARTESIAN_DIM;
}

int spn_convsp_forward(const THFloatTensor* locs_t, const THFloatTensor* data_t, 
    const THFloatTensor* weight_t, const THFloatTensor* bias_t, const float radius, 
    const THFloatTensor* kernel_size_t, const THFloatTensor* dilation_t, 
    const int dis_norm, const int kernel_fn, THFloatTensor* out_t)
{

    const float* locs = THFloatTensor_data(locs_t);
    const float* data = THFloatTensor_data(data_t);
    const float* weight = THFloatTensor_data(weight_t);
    const float* bias = THFloatTensor_data(bias_t); 
    const int batch_size = locs_t->size[0];
    const int N = locs_t->size[1];
    const int nchannels = data_t->size[2];
    const int ndims = locs_t->size[2];
    const int nkernels = weight_t->size[0];
    const int ncells = weight_t->size[2];
    const float* kernel_size = THFloatTensor_data(kernel_size_t);
    const float* dilation = THFloatTensor_data(dilation_t);
    float* out = THFloatTensor_data(out_t);

    return cpu_convsp(locs, data, weight, bias, batch_size, N, nchannels, ndims,
        nkernels, ncells, radius, kernel_size, dilation, dis_norm, kernel_fn, out, NULL, 
        NULL);
}

int spn_convsp_backward(const THFloatTensor* locs_t, const THFloatTensor* data_t, 
    const THFloatTensor* weight_t, const THFloatTensor* bias_t, const float radius, 
    const THFloatTensor* kernel_size_t, const THFloatTensor* dilation_t, 
    const int dis_norm, const int kernel_fn, THFloatTensor* out_t, 
    THFloatTensor* ddata_t, THFloatTensor* dweight_t)
{

    const float* locs = THFloatTensor_data(locs_t);
    const float* data = THFloatTensor_data(data_t);
    const float* weight = THFloatTensor_data(weight_t);
    const float* bias = THFloatTensor_data(bias_t); 
    float* ddata = THFloatTensor_data(ddata_t);
    float* dweight = THFloatTensor_data(dweight_t);
    const int batch_size = locs_t->size[0];
    const int N = locs_t->size[1];
    const int nchannels = data_t->size[2];
    const int ndims = locs_t->size[2];
    const int nkernels = weight_t->size[0];
    const int ncells = weight_t->size[2];
    const float* kernel_size = THFloatTensor_data(kernel_size_t);
    const float* dilation = THFloatTensor_data(dilation_t);
    float* out = THFloatTensor_data(out_t);

    return cpu_convsp(locs, data, weight, bias, batch_size, N, nchannels, ndims,
        nkernels, ncells, radius, kernel_size, dilation, dis_norm, kernel_fn, out, 
        ddata, dweight);
}

int cpu_convsp(const float* locs, const float* data, const float* weight, const float* bias, 
    const int batch_size, const int N, const int nchannels, const int ndims, 
    const int nkernels, const int ncells, const float radius, const float* kernel_size, 
    const float* dilation, const int dis_norm, const int kernel_fn, float* out, 
    float* ddata, float* dweight)
{
    int b, n;
    for(b = 0; b < batch_size; ++b)
    {
        for(n = 0; n < N; ++n)
        {
            compute_kernel_cells(locs, data, weight, bias, batch_size, N, 
                nchannels, ndims, nkernels, ncells, radius, kernel_size, dilation, 
                dis_norm, kernel_fn, out, b, n, 0, N, ddata, dweight);
        }
    }
    return 1;
}




int spn_convsdf_forward(const THFloatTensor* locs_t, const THFloatTensor* idxs_t, 
    const THFloatTensor* poses_t, const THFloatTensor* scales_t, 
    const THFloatTensor* sdfs_t, const THFloatTensor* sdf_offsets_t,
    const THFloatTensor* sdf_shapes_t, const THFloatTensor* weight_t, 
    const THFloatTensor* bias_t, const THFloatTensor* kernel_size_t, 
    const THFloatTensor* dilation_t, const float max_distance,
    THFloatTensor* out_t)
{

    const float* locs = THFloatTensor_data(locs_t);
    const float* idxs = THFloatTensor_data(idxs_t);
    const float* poses = THFloatTensor_data(poses_t);
    const float* scales = THFloatTensor_data(scales_t);
    const float* sdfs = THFloatTensor_data(sdfs_t);
    const float* sdf_offsets = THFloatTensor_data(sdf_offsets_t);
    const float* sdf_shapes = THFloatTensor_data(sdf_shapes_t);
    const float* weight = THFloatTensor_data(weight_t);
    const float* bias = THFloatTensor_data(bias_t); 
    const int batch_size = locs_t->size[0];
    const int N = locs_t->size[1];
    const int ndims = locs_t->size[2];
    const int M = idxs_t->size[1];
    const int pose_len = poses_t->size[2];
    const int nkernels = weight_t->size[0];
    const int ncells = weight_t->size[1];
    const float* kernel_size = THFloatTensor_data(kernel_size_t);
    const float* dilation = THFloatTensor_data(dilation_t);
    float* out = THFloatTensor_data(out_t);

    return cpu_convsdf(locs, batch_size, N, ndims, idxs, poses, scales, M, pose_len, sdfs, 
        sdf_offsets, sdf_shapes, weight, bias, nkernels, ncells, kernel_size, dilation, 
        max_distance, out, NULL);
}

int spn_convsdf_backward(const THFloatTensor* locs_t, const THFloatTensor* idxs_t, 
    const THFloatTensor* poses_t, const THFloatTensor* scales_t, const THFloatTensor* sdfs_t, 
    const THFloatTensor* sdf_offsets_t, const THFloatTensor* sdf_shapes_t, 
    const THFloatTensor* weight_t, const THFloatTensor* bias_t, 
    const THFloatTensor* kernel_size_t, const THFloatTensor* dilation_t, 
    const float max_distance, THFloatTensor* out_t, THFloatTensor* dweight_t)
{

    const float* locs = THFloatTensor_data(locs_t);
    const float* idxs = THFloatTensor_data(idxs_t);
    const float* poses = THFloatTensor_data(poses_t);
    const float* scales = THFloatTensor_data(scales_t);
    const float* sdfs = THFloatTensor_data(sdfs_t);
    const float* sdf_offsets = THFloatTensor_data(sdf_offsets_t);
    const float* sdf_shapes = THFloatTensor_data(sdf_shapes_t);
    const float* weight = THFloatTensor_data(weight_t);
    const float* bias = THFloatTensor_data(bias_t); 
    float* dweight = THFloatTensor_data(dweight_t); 
    const int batch_size = locs_t->size[0];
    const int N = locs_t->size[1];
    const int ndims = locs_t->size[2];
    const int M = idxs_t->size[1];
    const int pose_len = poses_t->size[2];
    const int nkernels = weight_t->size[0];
    const int ncells = weight_t->size[1];
    const float* kernel_size = THFloatTensor_data(kernel_size_t);
    const float* dilation = THFloatTensor_data(dilation_t);
    float* out = THFloatTensor_data(out_t);

    return cpu_convsdf(locs, batch_size, N, ndims, idxs, poses, scales, M, pose_len, sdfs, 
        sdf_offsets, sdf_shapes, weight, bias, nkernels, ncells, kernel_size, dilation, 
        max_distance, out, dweight);
}

int cpu_convsdf(const float* locs, const int batch_size, const int N, const int ndims, 
    const float* idxs, const float* poses, const float* scales, const int M, 
    const int pose_len, const float* sdfs, const float* sdf_offsets, const float* sdf_shapes, 
    const float* weight, const float* bias, const int nkernels, const int ncells, 
    const float* kernel_size, const float* dilation, const float max_distance, 
    float* out, float* dweight)
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