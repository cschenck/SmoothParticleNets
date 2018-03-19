#ifndef __gpu_kernels_h__
#define __gpu_kernels_h__
#ifdef __cplusplus
extern "C" {
#endif

int cuda_convsp(
		const float* locs, 
		const float* data, 
		const float* weight, 
		const float* bias, 
		const int batch_size, 
		const int N, 
		const int nchannels, 
		const int ndims, 
		const int nkernels, 
		const int ncells, 
		const float radius, 
		const float* kernel_size, 
		const float* dilation, 
		const int dis_norm, 
		const int kernel_fn, 
		float* out, 
		float* ddata, 
		float* dweight, 
		cudaStream_t stream, 
		const size_t nshared_device_mem);

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
    float* dweight, 
    cudaStream_t stream);

size_t GetSharedMemPerBlock(int device);

#ifdef __cplusplus
}
#endif

#endif