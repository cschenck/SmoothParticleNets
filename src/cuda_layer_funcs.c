
#include <stdio.h>

#include <TH/TH.h>


#ifdef WITH_CUDA
#include <THC/THC.h>

#include "gpu_kernels.h"

extern THCState *state;



int spnc_convsp_forward(THCudaTensor* locs_t, THCudaTensor* data_t, THCudaTensor* density_t, 
    THCudaTensor* weight_t, THCudaTensor* bias_t, float radius, 
    THCudaTensor* kernel_size_t, THCudaTensor* dilation_t, THCudaTensor* out_t)
{

    float* locs = THCudaTensor_data(state, locs_t);
    float* data = THCudaTensor_data(state, data_t);
    float* density = THCudaTensor_data(state, density_t);
    float* weight = THCudaTensor_data(state, weight_t);
    float* bias = THCudaTensor_data(state, bias_t);
    int batch_size = locs_t->size[0];
    int N = locs_t->size[1];
    int nchannels = data_t->size[1];
    int ndims = locs_t->size[2] - 1;
    int nkernels = weight_t->size[0];
    int ncells = weight_t->size[2];
    float* kernel_size = THCudaTensor_data(state, kernel_size_t);
    float* dilation = THCudaTensor_data(state, dilation_t);
    float* out = THCudaTensor_data(state, out_t);
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_convsp(locs, data, density, weight, bias, batch_size, N, nchannels, ndims,
        nkernels, ncells, radius, kernel_size, dilation, out, NULL, NULL, stream);

}

int spnc_convsp_backward(THCudaTensor* locs_t, THCudaTensor* data_t, THCudaTensor* density_t, 
    THCudaTensor* weight_t, THCudaTensor* bias_t, float radius, 
    THCudaTensor* kernel_size_t, THCudaTensor* dilation_t, THCudaTensor* out_t,
    THCudaTensor* ddata_t, THCudaTensor* dweight_t)
{
    float* locs = THCudaTensor_data(state, locs_t);
    float* data = THCudaTensor_data(state, data_t);
    float* density = THCudaTensor_data(state, density_t);
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
    float* kernel_size = THCudaTensor_data(state, kernel_size_t);
    float* dilation = THCudaTensor_data(state, dilation_t);
    float* out = THCudaTensor_data(state, out_t);
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_convsp(locs, data, density, weight, bias, batch_size, N, nchannels, ndims,
        nkernels, ncells, radius, kernel_size, dilation, out, ddata, dweight, stream);
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

#endif