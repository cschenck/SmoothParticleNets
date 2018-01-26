#ifndef __gpu_kernels_h__
#define __gpu_kernels_h__
#ifdef __cplusplus
extern "C" {
#endif

int cuda_convsp(float* locs, float* data, float* density, float* weight, float* bias, 
	int batch_size, int N, int nchannels, int ndims, int nkernels, int ncells, 
	float radius, float* kernel_size, float* dilation, float* out, float* ddata,
	float* dweight, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif