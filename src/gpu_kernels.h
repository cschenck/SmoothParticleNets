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

int cuda_particles2grid(float* points, 
					    float* data, 
					    float* density, 
					    int nparticles, 
					    int batch_size,
					    int data_dims, 
					    float grid_lowerx,
					    float grid_lowery,
					    float grid_lowerz,
						int grid_dimsx,
						int grid_dimsy,
						int grid_dimsz,
						float grid_stepsx,
						float grid_stepsy,
						float grid_stepsz,
						float* grid, 
						float radius, 
						cudaStream_t stream);


#ifdef __cplusplus
}
#endif

#endif