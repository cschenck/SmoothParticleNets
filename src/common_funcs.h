#ifndef __common_funcs_h__
#define __common_funcs_h__
#ifdef __cplusplus
extern "C" {
#endif

#ifdef CUDA
	#define DEVICE_FUNC __device__ inline
	#define GLOBAL_FUNC __global__ inline
#else
	#define DEVICE_FUNC inline
	#define GLOBAL_FUNC inline
#endif

#include <math.h>
#include <stdio.h>

#include "constants.h"

#ifndef CUDA
void atomicAdd(float* ptr, float value)
{
	*ptr += value;
}
#endif

DEVICE_FUNC
float kernel_w(float d, float radius)
{
	return 8.0f/powf(radius, 3)*(0.25f*powf(fmaxf(0.0f, radius - d), 3) - 
		powf(fmaxf(0.0f, radius/2.0f - d), 3));
}


DEVICE_FUNC
float dissqr(float* x, float* y, int ndims)
{
	float ret = 0.0f;
	int i;
	for(i = 0; i < ndims; ++i)
		ret += (x[i] - y[i])*(x[i] - y[i]);
	return ret;
}

DEVICE_FUNC
float fastroot(float x)
{
	if(x == 1.0f)
		return 1.0f;
	else if(x == 2.0f)
		return 1.41421f;
	else if(x == 3.0f)
		return 1.73205f;
	else
		return sqrtf(x);
}

DEVICE_FUNC
float lmaxf(float* x, int len)
{
	float ret = x[0];
	int i;
	for(i = 1; i < len; ++i)
	{
		if(x[i] > ret)
			ret = x[i];
	}
	return ret;
}

DEVICE_FUNC
int lmaxi(int* x, int len)
{
	int ret = x[0];
	int i;
	for(i = 1; i < len; ++i)
	{
		if(x[i] > ret)
			ret = x[i];
	}
	return ret;
}


/**
Function that that computes the partial values for the kernel cells for a given particle.
Given a particle index in a specific batch, this function loops through the given range
of particle indices, adding their contribution to each of the kernel cells around the
given particle. Inputs are:
	-locs: (batch_size X N X ndimss+1) the cartesian coordinates of all the particles.
	-data: (batch_size X nchannels X N) the features associated with each particle.
	-density: (batch_size X N) the density at each particle.
	-weight: (nkernels X nchannels X ncells) the kernel weights.
	-bias: (nkernels) the kernel biases.
	-batch_size: the size of the batch.
	-N: the number of particles in each batch.
	-nchannels: the number of features per particle.
	-ndims: the cardinality of the cartesian coordinate space.
	-nkernels: the number of convolution kernels.
	-ncells: the number of cells in each kernel (this is the product of all the values
			 in kernel_size).
	-radius: the radius to use when doing neighbordhood lookups around a query point.
	-kernel_size: (ndims) the number of kernel cells in each dimension.
	-dilation: (ndims) the size of a signal kernel cell in each dimension.
	-out: (batch_size X nkernels X N) the partially computed output values for
		  each particle.
	-b: the batch index of the given particle.
	-n: the particle index of the given particle.
	-start: the particle index to start at when computing neighborhood lookups (inclusive).
	-end: the particle index to end at when computing neighborhood lookups (exclusive).
	-ddata: [Optional] (batch_size X nchannels X N) if not NULL, then this function will
			compute the derivative. This assumes that out is filled with the derivative
			of the output of this layer wrt some loss, and that this is initially
			filled with all 0s. The derivative wrt the data is stored here.
	-dweight [Optional] (nkernels X nchannels X ncells) similar to ddata, if not NULL
			 then the derivative wrt the weights is computed. It is expected that ddata
			 and dweight will either be both NULL (forward computation) or both not
			 NULL (backward computation).
**/
DEVICE_FUNC
void compute_kernel_cells(float* locs, float* data, float* density, float* weight, 
	float* bias, int batch_size, int N, int nchannels, int ndims, int nkernels, int ncells,
	float radius, float* kernel_size, float* dilation, float* out, int b, int n, int start, 
	int end, float* ddata, float* dweight)
{
	int idxs[MAX_CARTESIAN_DIM];
	float* r = locs + (b*N + n)*(ndims + 1);
	int backward = ((ddata != NULL) || (dweight != NULL));
	float* out_ptr = out + b*nkernels*N + n;

	int j;
	for(j = start; j < end && j < N; ++j)
	{
		float* r2 = locs + (b*N + j)*(ndims + 1);
		float d = dissqr(r, r2, ndims);
		float dd = fastroot(ndims);
		float maxdil = lmaxf(dilation, ndims);
		int maxkern = (int)lmaxf(kernel_size, ndims)/2;
		float nr = radius + maxkern*maxdil*dd;
		if(d > nr*nr)
			continue;

		
		int k;
		for(k = 0; k < ndims; ++k)
			idxs[k] = 0;
		
		
		float* data_ptr = data + b*nchannels*N + j;
		float* ddata_ptr = ddata + b*nchannels*N + j;
		int kernel_idx;
		for(kernel_idx = 0; idxs[ndims-1] < kernel_size[ndims-1]; ++kernel_idx)
		{
			d = 0.0f;
			for(k = 0; k < ndims; ++k)
			{
				nr = r[k] + (idxs[k] - ((int)kernel_size[k])/2)*dilation[k] - r2[k];
				d += nr*nr;
			}
			if(d < radius*radius)
			{
				d = sqrtf(d);
				int outk, ink;
				for(outk = 0; outk < nkernels; ++outk)
				{
					for(ink = 0; ink < nchannels; ++ink)
					{
						// The full pointer computation for reference. Common 
						// terms have been taken out of the loop for
						// efficiency.
						// out + 
						//   b*nkernels*N + 
						//   outk*N +
						//   n 
						// data + b*nchannels*N + ink*N + j
						if(backward)
						{
							if(ddata != NULL)
								atomicAdd(ddata_ptr + ink*N, 
									(*(out_ptr + outk*N))*
									weight[outk*nchannels*ncells + ink*ncells + kernel_idx]*
									1.0f/(r2[ndims]*density[b*N + j])*
									kernel_w(d, radius));
							if(dweight != NULL)
								atomicAdd(dweight + outk*nchannels*ncells + ink*ncells + kernel_idx, 
									(*(out_ptr + outk*N))*
									1.0f/(r2[ndims]*density[b*N + j])*
									(*(data_ptr + ink*N))*
									kernel_w(d, radius));
						}
						else
						{
							atomicAdd(out_ptr + outk*N, 
								weight[outk*nchannels*ncells + ink*ncells + kernel_idx]*
								1.0f/(r2[ndims]*density[b*N + j])*
								(*(data_ptr + ink*N))*
								kernel_w(d, radius));
						}
					}
				}
			}
			++idxs[0];
			for(k = 0; k < ndims - 1 && idxs[k] >= kernel_size[k]; ++k)
			{
				idxs[k] = 0;
				++idxs[k+1];
			}
		}
	}
}

#ifdef __cplusplus
}
#endif
#endif