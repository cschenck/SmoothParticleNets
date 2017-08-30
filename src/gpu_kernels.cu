
#ifdef __cplusplus
extern "C" {
#endif


#include <math.h>
#include <stdio.h>

#include "gpu_kernels.h"


typedef struct 
{
	int d[MAX_TENSOR_DIM];
} TensorShape;

__global__
void kernel_assign_from_locs(float* locs, float* data, int batch_size, int nlocs, int dlocs, TensorShape dim_sizes, 
	int elsize, float* out)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < batch_size*nlocs*elsize; i += stride)
  {
  	int b = i/(nlocs*elsize);
  	int n = (i%(nlocs*elsize))/elsize;
  	int e = i%elsize;
  	float* fidxs = locs + b*nlocs*dlocs + n*dlocs;
  	int s = elsize;
  	int di = e;
  	bool inbounds = true;
  	for(int j = dlocs - 1; j >= 0 && inbounds; --j)
  	{
  		if(fidxs[j] < 0 || fidxs[j] >= dim_sizes.d[j])
  			inbounds = false;
  		di += fidxs[j]*s;
  		s *= dim_sizes.d[j];
  	}
  	di += b*s;
  	if(inbounds)
  		out[i] = data[di];
  	else
  		out[i] = 0.0;
  }
}
int cuda_assign_from_locs(float* locs, float* data, int batch_size, int nlocs, int dlocs, const int* dim_sizes,
	int ddata, const int* data_dims, float* out, cudaStream_t stream)
{
	TensorShape _dim_sizes;
	for(int i = 0; i < dlocs; ++i)
		_dim_sizes.d[i] = dim_sizes[i];
	int elsize = 1;
	for(int i = 0; i < ddata; ++i)
		elsize *= data_dims[i];

	int nops = batch_size*nlocs*elsize;
    int numBlocks = ceil(nops * (1.0/256));
    dim3 blocks(numBlocks);
    dim3 threads(256);

	kernel_assign_from_locs<<<blocks, threads, 0, stream>>>(locs, data, batch_size, nlocs, dlocs, _dim_sizes, 
		elsize, out);
	// check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
	printf("error in cuda_assign_from_locs: %s\n", cudaGetErrorString(err));
		return 0;
	}
	return 1;
}


__global__
void kernel_add_to_locs(float* locs, float* data, int batch_size, int nlocs, int dlocs, TensorShape dim_sizes, 
	int elsize, float* out)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < batch_size*nlocs*elsize; i += stride)
  {
  	int b = i/(nlocs*elsize);
  	int n = (i%(nlocs*elsize))/elsize;
  	int e = i%elsize;
  	float* fidxs = locs + b*nlocs*dlocs + n*dlocs;
  	int s = elsize;
  	int di = e;
  	bool inbounds = true;
  	for(int j = dlocs - 1; j >= 0 && inbounds; --j)
  	{
  		if(fidxs[j] < 0 || fidxs[j] >= dim_sizes.d[j])
  			inbounds = false;
  		di += fidxs[j]*s;
  		s *= dim_sizes.d[j];
  	}
  	di += b*s;
  	if(inbounds)
  	{
  		atomicAdd(out + di, data[b*nlocs*elsize + n*elsize + e]);
  	}
  }
}
int cuda_add_to_locs(float* locs, float* data, int batch_size, int nlocs, int dlocs, const int* dim_sizes,
	int ddata, const int* data_dims, float* out, cudaStream_t stream)
{
	TensorShape _dim_sizes;
	for(int i = 0; i < dlocs; ++i)
		_dim_sizes.d[i] = dim_sizes[i];
	int elsize = 1;
	for(int i = 0; i < ddata; ++i)
		elsize *= data_dims[i];

	int nops = batch_size*nlocs*elsize;
    int numBlocks = ceil(nops * (1.0/256));
    dim3 blocks(numBlocks);
    dim3 threads(256);

	kernel_add_to_locs<<<blocks, threads, 0, stream>>>(locs, data, batch_size, nlocs, dlocs, _dim_sizes, 
		elsize, out);
	// check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
	printf("error in cuda_add_to_locs: %s\n", cudaGetErrorString(err));
		return 0;
	}
	return 1;
}




#ifdef __cplusplus
}
#endif
