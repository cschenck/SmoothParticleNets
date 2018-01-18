
#ifdef __cplusplus
extern "C" {
#endif

#include <float.h>
#include <math.h>
#include <stdio.h>

#include "gpu_kernels.h"

#define MAX_BLOCKS 65535


typedef struct 
{
	int d[MAX_TENSOR_DIM];
} TensorShape;


/* UTIL FUNCTIONS */
__device__ __host__
float4 normalize_quaternion(float4 q)
{
    float m = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
    q.x /= m;
    q.y /= m;
    q.z /= m;
    q.w /= m;
    return q;
}

__device__ __host__
float4 quaternion_conjugate(float4 quat)
{
    float4 ret = quat;
    ret.x *= -1.0;
    ret.y *= -1.0;
    ret.z *= -1.0;
    return ret;
}

__device__ __host__
float4 quaternion_mult(float4 q1, float4 q2)
{
    float4 ret;
    ret.w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z;
    ret.x = q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y;
    ret.y = q1.w*q2.y + q1.y*q2.w + q1.z*q2.x - q1.x*q2.z;
    ret.z = q1.w*q2.z + q1.z*q2.w + q1.x*q2.y - q1.y*q2.x;
    return ret;
}

__device__ __host__
float trilinear_interp(float* grid, float3 grid_lower, int3 grid_dims, 
	float3 grid_steps, int data_dims, int data_dim, float4 point)
{
	float px = point.x;
	float py = point.y;
	float pz = point.z;
	float pi = (px - grid_lower.x)/grid_steps.x - 0.5;
	float pj = (py - grid_lower.y)/grid_steps.y - 0.5;
	float pk = (pz - grid_lower.z)/grid_steps.z - 0.5;
	float ii = pi - floorf(pi);
	float jj = pj - floorf(pj);
	float kk = pk - floorf(pk);
	float dd = 0;
	for(int di = 0; di < 2; ++di)
	{
		for(int dj = 0; dj < 2; ++dj)
		{
			for(int dk = 0; dk < 2; ++dk)
			{
				int ci = (int)fmaxf(0, fminf(grid_dims.x - 1, floorf(pi + di)));
				int cj = (int)fmaxf(0, fminf(grid_dims.y - 1, floorf(pj + dj)));
				int ck = (int)fmaxf(0, fminf(grid_dims.z - 1, floorf(pk + dk)));
				float v = grid[ci*grid_dims.y*grid_dims.z*data_dims + 
							   cj*grid_dims.z*data_dims + 
							   ck*data_dims + 
							   data_dim];
				dd += v*(di ? ii : 1 - ii)*(dj ? jj : 1 - jj)*(dk ? kk : 1 - kk);
			}
		}
	}
	return dd;
}

__global__
void kernel_cudaFloatMemset(float* ptr, float value, int nvals)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < nvals; i += stride)
	{
		ptr[i] = value;
	}
}
void cudaFloatMemset(float* ptr, float value, int nvals, cudaStream_t stream)
{
	int nops = nvals/256;
    int numBlocks = ceil(nops * (1.0/256));
    dim3 blocks(numBlocks);
    dim3 threads(256);
    kernel_cudaFloatMemset<<<blocks, threads, 0, stream>>>(ptr, value, nvals);
}

__device__ 
float atomicMin(float *addr, float value)
{
    float old = *addr;
    float assumed;
    do
    {
    	if(old <= value) 
    		return old;
        assumed = old;
        old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));
    }while(old != assumed);
    return old;
}

__device__
float kernel_w(float3 x, float radius)
{
	float d = sqrtf(x.x*x.x + x.y*x.y + x.z*x.z);
	return 8.0f/powf(radius, 3)*(0.25f*powf(fmaxf(0.0f, radius - d), 3) - 
		powf(fmaxf(0.0f, radius/2.0f - d), 3));
}




/* Layer functions */

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
	for (int i = index; i < (long long)bs*diam.x*diam.y*diam.z*nparticles; i += stride)
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
		// float d = (cx - px)*(cx - px) + (cy - py)*(cy - py) + (cz - pz)*(cz - pz);
		// if(d >= radius*radius)
		// 	continue;
		// d = sqrtf(d)/radius*2;
		// float w;
		// if(d < 1.0)
		// 	w = 1.0f/M_PI*((2 - d)*(2 - d)*(2 - d)/4 - (1 - d)*(1 - d)*(1 - d));
		// else
		// 	w = 1.0f/M_PI*((2 - d)*(2 - d)*(2 - d)/4);
		// We're going to compute the element of the SPH equation contributed by
		// particle p. First compute the kernel function based on the distance from
		// p to this location.
		float w = kernel_w(make_float3(cx - px, cy - py, cz - pz), radius);
		// Now multiply in the mass and density since those will remain constant
		// while we loop over the data values. DON'T FORGET p.w IS THE INVERSE MASS!!!
		w /= (p.w*density[ii]);

		float* dd = data + ii*data_dims;
		// Finally, loop over the data values associated with p and sum the
		// contribution to this location.
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
	cudaMemset(grid, 0, sizeof(float)*batch_size*grid_dimsx*grid_dimsy*grid_dimsz*data_dims);
	float3 grid_lower = make_float3(grid_lowerx, grid_lowery, grid_lowerz);
	int3 grid_dims = make_int3(grid_dimsx, grid_dimsy, grid_dimsz);
	float3 grid_steps = make_float3(grid_stepsx, grid_stepsy, grid_stepsz);
	int3 rsteps = make_int3(ceil(radius/grid_stepsx),
							ceil(radius/grid_stepsy),
							ceil(radius/grid_stepsz));
	float4* particles = (float4*)points;

	long long nops = (long long)batch_size*(2*rsteps.x + 1)*(2*rsteps.y + 1)*
						(2*rsteps.z + 1)*nparticles;
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



__global__
void kernel_trilinear_grid2particles(float* grid, int batch_size, float3 grid_lower, int3 grid_dims, 
	float3 grid_steps, int data_dims, float4* particles, int nparticles, float* data)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < batch_size*nparticles*data_dims; i += stride)
	{
		const int b = i/(nparticles*data_dims);
		const float4& p = particles[i/data_dims];
		const int d = i%data_dims;

		data[i] = trilinear_interp(grid + b*grid_dims.x*grid_dims.y*grid_dims.z*data_dims, 
			grid_lower, grid_dims, grid_steps, data_dims, d, p);
	}
}
int cuda_grid2particles(float* grid, 
					     int batch_size,
					     float grid_lowerx,
					     float grid_lowery,
					     float grid_lowerz,
						 int grid_dimsx,
						 int grid_dimsy,
						 int grid_dimsz,
						 float grid_stepsx,
						 float grid_stepsy,
						 float grid_stepsz,
						 int data_dims, 
						 float* points, 
						 int nparticles,
					     float* data,  
						 cudaStream_t stream)
{
	float3 grid_lower = make_float3(grid_lowerx, grid_lowery, grid_lowerz);
	int3 grid_dims = make_int3(grid_dimsx, grid_dimsy, grid_dimsz);
	float3 grid_steps = make_float3(grid_stepsx, grid_stepsy, grid_stepsz);
	float4* particles = (float4*)points;

	int nops = batch_size*nparticles*data_dims;
    int numBlocks = min(MAX_BLOCKS, (int)ceil(nops/256.0f));
    dim3 blocks(numBlocks);
    dim3 threads(256);

    kernel_trilinear_grid2particles<<<blocks, threads, 0, stream>>>(grid, batch_size, grid_lower,
    	grid_dims, grid_steps, data_dims, particles, nparticles, data);
    cudaDeviceSynchronize();
    // check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in cuda_grid2particles: %s\n", cudaGetErrorString(err));
		return 0;
	}
    return 1;
}


__global__
void kernel_backward_grid2particles(float4* particles, float* ddata, int batch_size, int nparticles, int data_dims,
    float* dgrid, float3 grid_lower, int3 grid_dims, float3 grid_steps)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < batch_size*nparticles*data_dims; i += stride)
    {
        const int b = i/(nparticles*data_dims);
        const float4& p = particles[i/data_dims];
        const int d = i%data_dims;
        float px = p.x;
        float py = p.y;
        float pz = p.z;
        float pi = (px - grid_lower.x)/grid_steps.x - 0.5;
        float pj = (py - grid_lower.y)/grid_steps.y - 0.5;
        float pk = (pz - grid_lower.z)/grid_steps.z - 0.5;
        float ii = pi - floorf(pi);
        float jj = pj - floorf(pj);
        float kk = pk - floorf(pk);
        for(int di = 0; di < 2; ++di)
        {
            for(int dj = 0; dj < 2; ++dj)
            {
                for(int dk = 0; dk < 2; ++dk)
                {
                    int ci = (int)fmaxf(0, fminf(grid_dims.x - 1, floorf(pi + di)));
					int cj = (int)fmaxf(0, fminf(grid_dims.y - 1, floorf(pj + dj)));
					int ck = (int)fmaxf(0, fminf(grid_dims.z - 1, floorf(pk + dk)));
                    float* v = dgrid + b*grid_dims.x*grid_dims.y*grid_dims.z*data_dims +
                                      ci*grid_dims.y*grid_dims.z*data_dims + 
                                      cj*grid_dims.z*data_dims + 
                                      ck*data_dims + 
                                      d;
                    atomicAdd(v, ddata[i]*(di ? ii : 1 - ii)*(dj ? jj : 1 - jj)*(dk ? kk : 1 - kk));
                }
            }
        }
    }
}
int cuda_grid2particles_backward(float* points, float* ddata, int batch_size, int nparticles, int data_dims,
    float* dgrid, float grid_lowerx, float grid_lowery, float grid_lowerz, int grid_dimsx, int grid_dimsy,
    int grid_dimsz, float grid_stepsx, float grid_stepsy, float grid_stepsz, cudaStream_t stream)
{
    cudaMemset(dgrid, 0, sizeof(float)*batch_size*grid_dimsx*grid_dimsy*grid_dimsz*data_dims);
    float3 grid_lower = make_float3(grid_lowerx, grid_lowery, grid_lowerz);
    int3 grid_dims = make_int3(grid_dimsx, grid_dimsy, grid_dimsz);
    float3 grid_steps = make_float3(grid_stepsx, grid_stepsy, grid_stepsz);
    float4* particles = (float4*)points;

    int nops = batch_size*nparticles*data_dims;
    int numBlocks = min(MAX_BLOCKS, (int)ceil(nops/256.0f));
    dim3 blocks(numBlocks);
    dim3 threads(256);

    kernel_backward_grid2particles<<<blocks, threads, 0, stream>>>(particles, ddata, batch_size, nparticles, 
        data_dims, dgrid, grid_lower, grid_dims, grid_steps);
    cudaDeviceSynchronize();
    // check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in cuda_grid2particles_backward: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return 1;
}



__global__
void kernel_sdfs2grid(float** sdfs, int3** sdf_dims, float* sdf_poses, float3* sdf_widths, int batch_size, 
	int nsdfs, float* grid, float3 grid_lower, int3 grid_dims, float3 grid_steps)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < batch_size*nsdfs*grid_dims.x*grid_dims.y*grid_dims.z; i += stride)
    {
        // First get the batch number (b), the sdf number (s), and the 3D grid indices (ci, cj, ck).
        int ii = i;
        int b = ii/(nsdfs*grid_dims.x*grid_dims.y*grid_dims.z);
        ii -= b*(nsdfs*grid_dims.x*grid_dims.y*grid_dims.z);
        int s = ii/(grid_dims.x*grid_dims.y*grid_dims.z);
        int sb = s + b*nsdfs;
        ii -= s*(grid_dims.x*grid_dims.y*grid_dims.z);
        int ci = ii/(grid_dims.y*grid_dims.z);
        ii -= ci*(grid_dims.y*grid_dims.z);
        int cj = ii/grid_dims.z;
        int ck = ii%grid_dims.z;
        if(sdfs[sb] == NULL)
        	continue;

        // Now convert the 3D grid indices to xyz coordinates in 3D space.
        float4 pt;
        pt.x = grid_lower.x + (ci + 0.5)*grid_steps.x;
        pt.y = grid_lower.y + (cj + 0.5)*grid_steps.y;
        pt.z = grid_lower.z + (ck + 0.5)*grid_steps.z;

        // Now transform the 3D coordinates into the sdf's coordinate frame.
        float3 loc = *((float3*)(sdf_poses + sb*7));
        float4 rot = *((float4*)(sdf_poses + sb*7 + 3));
        rot = normalize_quaternion(rot);
        pt.x -= loc.x;
        pt.y -= loc.y;
        pt.z -= loc.z;
        pt = quaternion_mult(quaternion_mult(quaternion_conjugate(rot), pt), rot);

        // Now use trilinear interp to get the value at the 3D coordinates.
        float3 sdf_steps = make_float3(sdf_widths[sb].x/sdf_dims[sb]->x, sdf_widths[sb].y/sdf_dims[sb]->y,
            sdf_widths[sb].z/sdf_dims[sb]->z);
        if(pt.x >= 0.5*sdf_steps.x && pt.x <= sdf_dims[sb]->x*sdf_steps.x - 0.5 &&
           pt.y >= 0.5*sdf_steps.y && pt.y <= sdf_dims[sb]->y*sdf_steps.y - 0.5 &&
           pt.z >= 0.5*sdf_steps.z && pt.z <= sdf_dims[sb]->z*sdf_steps.z - 0.5)
        {
        	float v = trilinear_interp(sdfs[sb], make_float3(0, 0, 0), *sdf_dims[sb], sdf_steps, 
        		1, 0, pt);
        	atomicMin(grid + b*grid_dims.x*grid_dims.y*grid_dims.z +
		                     ci*grid_dims.y*grid_dims.z + 
		                     cj*grid_dims.z + 
		                     ck, 
	                   v);
    	}
    }
}
int cuda_forward_sdfs2grid(float* sdfs, int* sdf_dims, int stride_per_sdf, int nsdfs, int* sdf_indices,
	int batch_size, int nsdf_indices, float* sdf_poses,
	float* sdf_widths, float* grid, float grid_lowerx, float grid_lowery, float grid_lowerz, int grid_dimsx, 
	int grid_dimsy, int grid_dimsz, float grid_stepsx, float grid_stepsy, float grid_stepsz, 
	cudaStream_t stream)
{
	cudaFloatMemset(grid, FLT_MAX, sizeof(float)*batch_size*grid_dimsx*grid_dimsy*grid_dimsz, stream);
    float3 grid_lower = make_float3(grid_lowerx, grid_lowery, grid_lowerz);
    int3 grid_dims = make_int3(grid_dimsx, grid_dimsy, grid_dimsz);
    float3 grid_steps = make_float3(grid_stepsx, grid_stepsy, grid_stepsz);
    float3* sdf_widths3 = (float3*)sdf_widths;

    float** psdfs;
    int3** psdf_dims;
    cudaMalloc(&psdfs, sizeof(float*)*batch_size*nsdf_indices);
    cudaMalloc(&psdf_dims, sizeof(int3*)*batch_size*nsdf_indices);
    for(int i = 0; i < nsdf_indices; ++i)
    {
    	int idx = sdf_indices[i];
    	if(idx >= 0)
    	{
    		psdfs[i] = sdfs + idx*stride_per_sdf;
    		psdf_dims[i] = (int3*)(sdf_dims + idx*3);
    	}
    	else
    	{
    		psdfs[i] = NULL;
    		psdf_dims[i] = NULL;
    	}
    }

    int nops = batch_size*nsdf_indices*grid_dims.x*grid_dims.y*grid_dims.z;
    int numBlocks = min(MAX_BLOCKS, (int)ceil(nops/256.0f));
    dim3 blocks(numBlocks);
    dim3 threads(256);

    kernel_sdfs2grid<<<blocks, threads, 0, stream>>>(psdfs, psdf_dims, sdf_poses, sdf_widths3, batch_size,
    	nsdf_indices, grid, grid_lower, grid_dims, grid_steps);
    cudaDeviceSynchronize();
    // check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in cuda_forward_sdfs2grid: %s\n", cudaGetErrorString(err));
        return 0;
    }

    cudaFree(psdfs);
    cudaFree(psdf_dims);

    return 1;
}



























/* TEST CODE */
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
