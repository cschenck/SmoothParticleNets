
#ifdef __cplusplus
extern "C" {
#endif


#include <math.h>
#include <stdio.h>

#include "gpu_kernels.h"

#define MAX_BLOCKS 65535


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
	cudaDeviceSynchronize();
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
	cudaDeviceSynchronize();
	// check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
	printf("error in cuda_add_to_locs: %s\n", cudaGetErrorString(err));
		return 0;
	}
	return 1;
}


__global__
void kernel_sum_particle2grid_contributions(float4* points, float* data, float* density, int nparticles, 
	int batch_size, int data_dims, float3 grid_lower, int3 grid_dims, float3 grid_steps, float* grid, 
	float radius, int3 rsteps)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int3 diam = make_int3(2*rsteps.x + 1, 2*rsteps.y + 1, 2*rsteps.z + 1);
	int bs = batch_size;
	for (int i = index; i < bs*diam.x*diam.y*diam.z*nparticles; i += stride)
	{
		int b = i/(diam.x*diam.y*diam.z*nparticles);
		int ii = i/(diam.x*diam.y*diam.z);
		const float4& p = points[ii];
		float px = p.x;
		float py = p.y;
		float pz = p.z;
		int pi = (int)floorf((px - grid_lower.x)/grid_steps.x);
		int pj = (int)floorf((py - grid_lower.y)/grid_steps.y);
		int pk = (int)floorf((pz - grid_lower.z)/grid_steps.z);
		int remainder = i%(diam.x*diam.y*diam.z);
		int ci = remainder/(diam.y*diam.z);
		int cj = (remainder - ci*diam.y*diam.z)/(diam.z);
		int ck = remainder%(diam.z);
		ci = pi + ci - rsteps.x;
		cj = pj + cj - rsteps.y;
		ck = pk + ck - rsteps.z;
		if(ci < 0 || ci >= grid_dims.x)
			continue;
		if(cj < 0 || cj >= grid_dims.y)
			continue;
		if(ck < 0 || ck >= grid_dims.z)
			continue;
		float cx = (ci + 0.5)*grid_steps.x + grid_lower.x;
		float cy = (cj + 0.5)*grid_steps.y + grid_lower.y;
		float cz = (ck + 0.5)*grid_steps.z + grid_lower.z;
		float d = (cx - px)*(cx - px) + (cy - py)*(cy - py) + (cz - pz)*(cz - pz);
		if(d > radius*radius)
			continue;
		d = sqrtf(d)/radius*2;
		float w;
		if(d < 1.0)
			w = 1.0f/M_PI*((2 - d)*(2 - d)*(2 - d)/4 - (1 - d)*(1 - d)*(1 - d));
		else
			w = 1.0f/M_PI*((2 - d)*(2 - d)*(2 - d)/4);
		w *= p.w/density[ii];

		float* dd = data + ii*data_dims;
		for(int j = 0; j < data_dims; ++j) 
		{
			// printf("cuda: [%d] (%d, %d, %d) %10f %10f (%d, %d, %d) %d\n", ii, ci, cj, ck, w, dd[j], 
			// 	grid_dims.x, grid_dims.y, grid_dims.z, data_dims);
			atomicAdd(grid + b*grid_dims.x*grid_dims.y*grid_dims.z*data_dims +
							 ci*grid_dims.y*grid_dims.z*data_dims + 
							 cj*grid_dims.z*data_dims + 
							 ck*data_dims + 
							 j, w*dd[j]); 
		}

		
	}
}
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
						cudaStream_t stream)
{
	cudaMemset(grid, 0, sizeof(float)*grid_dimsx*grid_dimsy*grid_dimsz*data_dims);
	float3 grid_lower = make_float3(grid_lowerx, grid_lowery, grid_lowerz);
	int3 grid_dims = make_int3(grid_dimsx, grid_dimsy, grid_dimsz);
	float3 grid_steps = make_float3(grid_stepsx, grid_stepsy, grid_stepsz);
	int3 rsteps = make_int3(ceil(grid_stepsx*grid_dimsx/radius),
							ceil(grid_stepsy*grid_dimsy/radius),
							ceil(grid_stepsz*grid_dimsz/radius));
	float4* particles = (float4*)points;

	int nops = (2*rsteps.x + 1)*(2*rsteps.y + 1)*(2*rsteps.z + 1)*nparticles;
    int numBlocks = min(MAX_BLOCKS, (int)ceil(nops/256.0f));
    dim3 blocks(numBlocks);
    dim3 threads(256);

    kernel_sum_particle2grid_contributions<<<blocks, threads, 0, stream>>>(particles, data, density, nparticles,
    	batch_size, data_dims, grid_lower, grid_dims, grid_steps, grid, radius, rsteps);
    cudaDeviceSynchronize();
    // check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in cuda_particles2grid: %s\n", cudaGetErrorString(err));
		return 0;
	}
    return 1;
}


void af(float* ptr, float value) {
	cudaMemcpy(ptr, &value, sizeof(float), cudaMemcpyHostToDevice);
}
float gf(float* ptr) {
	float ret;
	cudaMemcpy(&ret, ptr, sizeof(float), cudaMemcpyDeviceToHost);
	return ret;
}
int test_func(void)
{
	float radius = 1.8*2;

	float* points;
	cudaMalloc((void**)&points, sizeof(float)*4*4);
	af(points+0, 3);af(points+1, 1);af(points+2, 3);af(points+3, 1.0/10);
	af(points+4, 3);af(points+5, 5);af(points+6, 4);af(points+7, 1.0/10);
	af(points+8, 6);af(points+9, 2);af(points+10, 1);af(points+11, 1.0/10);
	af(points+12, 6);af(points+13, 6);af(points+14, 6);af(points+15, 1.0/10);

	float* data;
	cudaMalloc((void**)&data, sizeof(float)*4*2);
	af(data+0, 1);af(data+1, 100);
	af(data+2, 2);af(data+3, 700);
	af(data+4, 5);af(data+5, 50);
	af(data+6, 4.5);af(data+7, 679);

	float* density;
	cudaMalloc((void**)&density, sizeof(float)*4);
	for(int i = 0; i < 4; ++i)
	{
		for(int j = 0; j < 4; ++j)
		{
			float x1 = gf(points + i*4 + 0), y1 = gf(points + i*4 + 1), z1 = gf(points + i*4 + 2);
			float x2 = gf(points + j*4 + 0), y2 = gf(points + j*4 + 1), z2 = gf(points + j*4 + 2);
			float d = (x1 - x2)*(x1 - x2) +
					  (y1 - y2)*(y1 - y2) +
					  (z1 - z2)*(z1 - z2);
			if(d <= radius*radius)
			{
				d = sqrt(d)/radius*2;
				af(density + i, gf(points +j*4 + 3)*(pow(2 - d, 3)/4.0 - pow(1 - d, 3)*(d < 1 ? 1 : 0))/M_PI);
			}
		}
	}

	float* grid;
	cudaMalloc((void**)&grid, sizeof(float)*3*3*3*2);

	cudaStream_t stream = 0;

	cuda_particles2grid(points, 
					    data, 
					    density, 
					    4, 
					    1,
					    2, 
					    1,
					    1,
					    1,
						3,
						3,
						3,
						2,
						2,
						2,
						grid, 
						radius, 
						stream);

	return 1;
}


int main(int argc, char** argv)
{
	test_func();
	return 0;
}


#ifdef __cplusplus
}
#endif
