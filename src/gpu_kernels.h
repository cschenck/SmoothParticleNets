#ifndef __gpu_kernels_h__
#define __gpu_kernels_h__
#ifdef __cplusplus
extern "C" {
#endif

int cuda_convsp(float* locs, float* data, float* weight, float* bias, 
	int batch_size, int N, int nchannels, int ndims, int nkernels, int ncells, 
	float radius, float* kernel_size, float* dilation, int dis_norm, int kernel_fn, float* out, 
	float* ddata, float* dweight, cudaStream_t stream, size_t nshared_device_mem);

int cuda_convsdf(float* locs, int batch_size, int N, int ndims, float* idxs,
    float* poses, float* scales, int M, int pose_len, float* sdfs, float* sdf_offsets, 
    float* sdf_shapes, float* weight, float* bias, int nkernels, int ncells, 
    float* kernel_size, float* dilation, float max_distance, float* out, float* dweight, 
    cudaStream_t stream);

size_t GetSharedMemPerBlock(int device);

#ifdef __cplusplus
}
#endif

#endif