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
#include "kernel_constants.h"

#ifndef CUDA
void atomicAdd(float* ptr, float value)
{
	*ptr += value;
}
typedef struct
{
	float x, y, z, w;
} float4;
#endif

DEVICE_FUNC
float kernel_w(const float d, const float H, const int fn)
{
	if(d > H) return 0.0f;
	if(!VALIDATE_KERNEL_ID(fn))
	{
		printf("ERROR: Unknown kernel function for id %d. Returning -1.\n", fn);
		return -1;
	}
	return KERNEL_W(d, H, fn);
}


DEVICE_FUNC
float dissqr(const float* x, const float* y, const int ndims)
{
	float ret = 0.0f;
	int i;
	for(i = 0; i < ndims; ++i)
		ret += (x[i] - y[i])*(x[i] - y[i]);
	return ret;
}

DEVICE_FUNC
float fastroot(const float x)
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
float lmaxf(const float* x, const int len)
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
int lmaxi(const int* x, const int len)
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

DEVICE_FUNC
float4 quaternion_conjugate(const float4 quat)
{
    float4 ret = quat;
    ret.x *= -1.0;
    ret.y *= -1.0;
    ret.z *= -1.0;
    return ret;
}

DEVICE_FUNC
float4 quaternion_mult(const float4 q1, const float4 q2)
{
    float4 ret;
    ret.w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z;
    ret.x = q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y;
    ret.y = q1.w*q2.y + q1.y*q2.w + q1.z*q2.x - q1.x*q2.z;
    ret.z = q1.w*q2.z + q1.z*q2.w + q1.x*q2.y - q1.y*q2.x;
    return ret;
}

DEVICE_FUNC
void point_in_coordinate_frame(const float* point, const int ndims, 
	const float* translation, const float* rotation, float* out)
{
	float m, theta;
	float4 r, p;
	int i;
	for(i = 0; i < ndims; ++i)
		out[i] = point[i] - translation[i];
	switch(ndims)
	{
		case 1:
			break;
		case 2:
			m = sqrtf(out[0]*out[0] + out[1]*out[1]);
			theta = atan2f(out[1], out[0]) - rotation[0];
			out[0] = m*cosf(theta);
			out[1] = m*sinf(theta);
			break;
		case 3:
			r.x = rotation[0]; r.y = rotation[1]; r.z = rotation[2]; r.w = rotation[3];
			p.x = out[0]; p.y = out[1]; p.z = out[2];
			p.w = 0.0f;
			p = quaternion_mult(quaternion_conjugate(r), quaternion_mult(p, r));
			out[0] = p.x;
			out[1] = p.y;
			out[2] = p.z;
			break;
		default:
			printf("ERROR: Rotations in %d dimensional space are not supported!\n", ndims);
			break;
	}
}

DEVICE_FUNC
float rec_nlinear_interp(const float* grid, const float* grid_dims, const int ndims, 
						const float* point01, int* lowidx, const int curdim)
{
	int i, j, k;
	if(curdim == ndims)
	{
		const float* ptr = grid;
		for(i = 0; i < ndims; ++i)
		{
			k = lowidx[i];
			for(j = i + 1; j < ndims; ++j)
				k *= grid_dims[j];
			ptr += k;
		}
		return *ptr;
	}	
	else
	{
		float v1 = rec_nlinear_interp(grid, grid_dims, ndims, point01, lowidx, curdim + 1);
		lowidx[curdim] += 1;
		float v2 = rec_nlinear_interp(grid, grid_dims, ndims, point01, lowidx, curdim + 1);
		lowidx[curdim] -= 1;
		return (1 - point01[curdim])*v1 + point01[curdim]*v2;
	}
}

DEVICE_FUNC
float nlinear_interp(const float* grid, const float* grid_dims, const int ndims, 
	const float cell_size, const float* point)
{
	int lowidx[MAX_CARTESIAN_DIM];
	float point01[MAX_CARTESIAN_DIM];
	int i;
	for(i = 0; i < ndims; ++i)
	{
		point01[i] = point[i]/cell_size - 0.5;
		lowidx[i] = (int)point01[i];
		point01[i] = point01[i] - floorf(point01[i]);
	}
	return rec_nlinear_interp(grid, grid_dims, ndims, point01, lowidx, 0);
}


/**
Function that that computes the partial values for the kernel cells for a given particle.
Given a particle index in a specific batch, this function loops through the given range
of particle indices, adding their contribution to each of the kernel cells around the
given particle. Inputs are:
	-locs: (batch_size X N X ndims) the cartesian coordinates of all the particles.
	-data: (batch_size X N X nchannels) the features associated with each particle.
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
	-kernel_fn: the id of the kernel function to use for W in the SPH equation.
	-dis_norm: divide the SPH values by the distance to the point.
	-out: (batch_size X N X nkernels) the partially computed output values for
		  each particle.
	-b: the batch index of the given particle.
	-n: the particle index of the given particle.
	-start: the particle index to start at when computing neighborhood lookups (inclusive).
	-end: the particle index to end at when computing neighborhood lookups (exclusive).
	-ddata: [Optional] (batch_size X N X nchannels) if not NULL, then this function will
			compute the derivative. This assumes that out is filled with the derivative
			of the output of this layer wrt some loss, and that this is initially
			filled with all 0s. The derivative wrt the data is stored here.
	-dweight [Optional] (nkernels X nchannels X ncells) similar to ddata, if not NULL
			 then the derivative wrt the weights is computed. It is expected that ddata
			 and dweight will either be both NULL (forward computation) or both not
			 NULL (backward computation).
**/
DEVICE_FUNC
void compute_kernel_cells(
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
		const int b,
		const int n,
		int start,
		const int end,
		float* ddata, 
		float* dweight)
{
	int idxs[MAX_CARTESIAN_DIM];
	const float* r = locs + (b*N + n)*ndims;
	int backward = ((ddata != NULL) || (dweight != NULL));
	float* out_ptrn = out + b*nkernels*N + n*nkernels;
	const float* data_ptrn = data + b*nchannels*N + n*nchannels;
	float* ddata_ptrn = ddata + b*nchannels*N + n*nchannels;

	if(start < n)
		start = n;

	int j;
	for(j = start; j < end && j < N; ++j)
	{
		const float* r2 = locs + (b*N + j)*ndims;
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
				float norm = 1.0f;
				if(dis_norm && d > 0.0f)
					norm /= d;
				float* out_ptrj = out + b*nkernels*N + j*nkernels;
				const float* data_ptrj = data + b*nchannels*N + j*nchannels;
				float* ddata_ptrj = ddata + b*nchannels*N + j*nchannels;
				float kw = kernel_w(d, radius, kernel_fn);
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
						float weightnj = weight[outk*nchannels*ncells + ink*ncells + 
												kernel_idx];
						float weightjn = weight[outk*nchannels*ncells + ink*ncells + 
													(ncells - kernel_idx - 1)];
						
						if(backward)
						{
							if(ddata != NULL)
							{
								atomicAdd(ddata_ptrj + ink, 
									(*(out_ptrn + outk))*weightnj*kw*norm);
								if(j != n)
									atomicAdd(ddata_ptrn + ink, 
										(*(out_ptrj + outk))*weightjn*kw*norm);
							}
							if(dweight != NULL)
							{
								atomicAdd(dweight + outk*nchannels*ncells + 
										  ink*ncells + kernel_idx, 
									(*(out_ptrn + outk))*(*(data_ptrj + ink))*kw*norm);
								if(j != n)
									atomicAdd(dweight + outk*nchannels*ncells + ink*ncells + 
											  (ncells - kernel_idx - 1), 
										(*(out_ptrj + outk))*(*(data_ptrn + ink))*kw*norm);
							}
						}
						else
						{
							float f1 = weightnj*(*(data_ptrj + ink))*kw*norm;
							float g1 = weightjn*(*(data_ptrn + ink))*kw*norm;
							atomicAdd(out_ptrn + outk, f1);
							if(j != n)
								atomicAdd(out_ptrj + outk, g1);
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

/**
Function that that computes the partial values for the kernel cells for a query location.
Given a location index in a specific batch, this function loops through the given SDFs
indexed in idxs and computes the minimum value at each kernel location around each cell
and multiplies those values by the kernel weights for the given kernel index outk, and
stores them in out at that location's and kernel's index. The inputs are:
	-locs: (batch_size X N X ndims) the cartesian coordinates of each query location.
	-batch_size: the size of the batch.
	-N: the number of particles in each batch.
	-ndims: the cardinality of the cartesian coordinate space.
	-idxs: (batch_size x M) the indices in the global SDF list of each of the SDFs
			convolved on here. M is the maximum number of SDFs that may be in a scene,
			but there can be fewer. When there are fewer, set the indices in this array
			that are not needed to -1, and they will be ignored.
	-poses: (batch_size x M X pose_len) the pose of each of the M SDFs. Each row is one 
			pose, where the first ndims values are the translation and the remaining 
			values are the rotation. The form of the rotation will vary depending on ndims.
			Currently the following is supported:
				-1D: empty (no rotation)
				-2D: a single value in radians where 0 is pointing directly along the
					 +x axis and pi/2 is pointing directly along +y.
				-3D: a normalized quaternion of the format xyzw.
			No other dimensionalities for rotations are supported. The origin is 
			assumed to be the lower corner of the SDF grid.
	-scales: (batch_size x M) The scale of each SDF in idxs.
	-M: the number of SDFs.
	-pose_len: the length of each row in poses.
	-sdfs: (long array of bytes) the master list of all sdfs into which idxs indexes.
	-sdf_offsets: (K) the offset into sdfs for each sdf.
	-sdf_shapes: (K X ndims+1) the shape of each sdf in sdfs. The shape is expected to
				 be of the form x_1, x_2, ..., cellSize where x_n is the number of 
				 grid cells in the n-th dimension and cellSize is the size in real
				 coordinates of the side of a grid cell.
	-weight: (nkernels X ncells) the kernel weights.
	-bias: (nkernels) the kernel biases.
	-nkernels: the number of convolution kernels.
	-ncells: the number of cells in each kernel (this is the product of all the values
			 in kernel_size).
	-kernel_size: (ndims) the number of kernel cells in each dimension.
	-dilation: (ndims) the size of a signal kernel cell in each dimension.
	-max_distance: a cap on the maximum SDF value. It will be min(max_distance, SDF(r)).
	-out: (batch_size X N X nkernels) the partially computed output values for
		  each query location.
	-b: the batch index of the given query location.
	-n: the particle index of the given query location.
	-outk: the index of the output kernel to sum for.
	-dweight [Optional] (nkernels X ncells) if not NULL then the derivative wrt to
			 the weights is computed INSTEAD OF the partial sum. Assumes that out
			 contains the already computed sum.
	-isdf_cache: (M) a pre-allocated int cache for this function to use. This is passed
				 as an argument instead of being allocated in the function itself to
				 improve efficiency.
	-fsdf_cache: (M) a pre-allocated float cache for this function to use. This is passed
				 as an argument instead of being allocated in the function itself to
				 improve efficiency.
**/
DEVICE_FUNC
void compute_sdf_kernel_cells(
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
		const int b, 
		const int n, 
		const int outk, 
		float* dweight, 
		int* isdf_cache, 
		float* fsdf_cache)
{
	float r2[MAX_CARTESIAN_DIM];
	int backward = (dweight != NULL);

	int i;
	for(i = 0; i < M; ++i)
	{
		isdf_cache[i] = 1;
		fsdf_cache[i] = 0;
	}

	const float* r = locs + (b*N + n)*ndims;
	float dd = fastroot(ndims);
	float maxdil = lmaxf(dilation, ndims);
	int maxkern = (int)lmaxf(kernel_size, ndims)/2;
	float nr = maxkern*maxdil*dd;

	float biggest_minVal = max_distance;

	// The most expensive thing will be to evaluate every cell of the kernel in every
	// SDF, so let's see if we can rule out some SDFs by using only the center of the
	// kernel first.
	int m;
	for(m = 0; m < M; ++m)
	{
		int mm = (int)idxs[b*M + m];
		if(mm < 0)
		{
			isdf_cache[m] = 0;
			continue;
		}
		float cell_size = sdf_shapes[mm*(ndims + 1) + ndims]*scales[b*M + m];
		point_in_coordinate_frame(r, ndims, poses + b*M*pose_len + m*pose_len, 
			poses + b*M*pose_len + m*pose_len + ndims, r2);
		int inbounds = 1;
		for(i = 0; i < ndims && isdf_cache[m]; ++i)
		{
			// Is it possible for any kernel cell to ever fall within this SDF?
			if(r2[i] + nr < 0.5*cell_size || 
			   r2[i] - nr > (sdf_shapes[mm*(ndims + 1) + i] - 0.5)*cell_size)
				isdf_cache[m] = 0;
			// Check to see if this point (r2) falls inside the SDF so in the
			// next step we don't get an out of bounds error.
			if(r2[i] < 0.5*cell_size || 
			   r2[i] > (sdf_shapes[mm*(ndims + 1) + i] - 0.5)*cell_size)
				inbounds = 0;
		}
		if(!isdf_cache[m] || !inbounds)
			continue;
		float v = nlinear_interp(sdfs + (int)sdf_offsets[mm], sdf_shapes + mm*(ndims + 1), 
			ndims, cell_size, r2)*scales[b*M + m];
		fsdf_cache[m] = v;
		// TODO: This is causing issues with the tests. For now we won't use 
		// biggest_minVal to weed out SDFs.
		// if(v + nr < biggest_minVal)
		// 	biggest_minVal = v + nr;
	}
	for(m = 0; m < M; ++m)
	{
		if(isdf_cache[m] && fsdf_cache[m] - nr > biggest_minVal)
			isdf_cache[m] = 0;
	}

	// Okay, we've thrown out all the SDFs that are too far from the kernel to
	// make a difference, now iterate over the remainder and convolve the kernel.
	float* out_ptr = out + b*nkernels*N + n*nkernels + outk;
	if(!backward)
		*out_ptr = 0;
	int kidxs[MAX_CARTESIAN_DIM];
	float pt[MAX_CARTESIAN_DIM];
	int k;
	for(k = 0; k < ndims; ++k)
		kidxs[k] = 0;
	int kernel_idx;
	for(kernel_idx = 0; kidxs[ndims-1] < kernel_size[ndims-1]; ++kernel_idx)
	{
		for(i = 0; i < ndims; ++i)
			pt[i] = r[i] + (kidxs[i] - ((int)kernel_size[i]/2))*dilation[i];
		float smallest = max_distance;
		for(m = 0; m < M; ++m)
		{
			if(!isdf_cache[m])
				continue;
			int mm = (int)idxs[b*M + m];
			float cell_size = sdf_shapes[mm*(ndims + 1) + ndims]*scales[b*M + m];
			point_in_coordinate_frame(pt, ndims, poses + b*M*pose_len + m*pose_len, 
				poses + b*M*pose_len + m*pose_len + ndims, r2);
			int inbounds = 1;
			for(i = 0; i < ndims && inbounds; ++i)
			{
				if(r2[i] < 0.5*cell_size || 
				   r2[i] > (sdf_shapes[mm*(ndims + 1) + i] - 0.5)*cell_size)
					inbounds = 0;
			}
			if(!inbounds) continue;
			float v = nlinear_interp(sdfs + (int)sdf_offsets[mm], sdf_shapes + mm*(ndims + 1), 
				ndims, cell_size, r2)*scales[b*M + m];
			if(v < smallest)
				smallest = v;
		}
		if(backward)
			atomicAdd(dweight + outk*ncells + kernel_idx, smallest*(*out_ptr));
		else
			*out_ptr += weight[outk*ncells + kernel_idx]*smallest;
		++kidxs[0];
		for(k = 0; k < ndims - 1 && kidxs[k] >= kernel_size[k]; ++k)
		{
			kidxs[k] = 0;
			++kidxs[k+1];
		}
	}
	if(!backward)
		*out_ptr += bias[outk];

}

#ifdef __cplusplus
}
#endif
#endif