#ifndef __gpu_kernels_h__
#define __gpu_kernels_h__
#ifdef __cplusplus
extern "C" {
#endif

#define MAX_TENSOR_DIM 20

int cuda_assign_from_locs(float* locs, float* data, int batch_size, int nlocs, int dlocs, const int* dim_sizes,
	int ddata, const int* data_dims, float* out, cudaStream_t stream);

int cuda_add_to_locs(float* locs, float* data, int batch_size, int nlocs, int dlocs, const int* dim_sizes,
	int ddata, const int* data_dims, float* out, cudaStream_t stream);


#ifdef __cplusplus
}
#endif

#endif