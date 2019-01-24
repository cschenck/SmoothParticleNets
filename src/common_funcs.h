#ifndef __common_funcs_h__
#define __common_funcs_h__
#ifdef __cplusplus
extern "C" {
#endif

#ifdef CUDA
	#define DEVICE_FUNC __device__ inline
	#define GLOBAL_FUNC __device__ __host__ inline
#else
	#define DEVICE_FUNC inline
	#define GLOBAL_FUNC inline
#endif

#include <math.h>
#include <stdio.h>

#include "constants.h"
#include "kernel_constants.h"

#define MAX(a, b) (a > b ? a : b)
#define MIN(a, b) (a < b ? a : b)

#ifdef CUDA
	
DEVICE_FUNC 
float atomicMax(float *addr, float value)
{
    float old = *addr;
    float assumed;
    do
    {
    	if(old >= value) 
    		return old;
        assumed = old;
        old = __uint_as_float(atomicCAS((unsigned int*)addr, __float_as_uint(assumed), __float_as_uint(value)));
    }while(old != assumed);
    return old;
}

#else

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
float kernel_dw(const float d, const float H, const int fn)
{
	if(d > H) return 0.0f;
	if(!VALIDATE_KERNEL_ID(fn))
	{
		printf("ERROR: Unknown kernel function for id %d. Returning -1.\n", fn);
		return -1;
	}
	return KERNEL_DW(d, H, fn);
}


DEVICE_FUNC
void swapf(float* a, float* b, int size)
{
	float f;
	int i;
	for(i = 0; i < size; ++i)
	{
		f = a[i];
		a[i] = b[i];
		b[i] = f;
	}
}


GLOBAL_FUNC
int loc2grid(float coord, float low_coord, float cellEdge)
{
	int ret = (coord - low_coord)/cellEdge;
	if(ret >= 0)
		return ret;
	else
		return 0;
}


GLOBAL_FUNC
int partial_grid_hash(int grid_coord, const float* grid_dims, int dim, int max_dim)
{
	if(grid_coord >= grid_dims[dim])
		grid_coord = grid_dims[dim] - 1;
	else if(grid_coord < 0)
		grid_coord = 0;
	int dd;
	int c = grid_coord;
    for(dd = dim + 1; dd < max_dim; ++dd)
        c *= grid_dims[dd];
    return c;
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
float4 sub_float4(const float4 f1, const float4 f2)
{
	float4 ret;
	ret.x = f1.x - f2.x;
	ret.y = f1.y - f2.y;
	ret.z = f1.z - f2.z;
	ret.w = f1.w - f2.w;
	return ret;
}

DEVICE_FUNC
void rotate_point(float* point, const int ndims, const float* rotation, int reverse)
{
	float m, theta;
	float4 r, p;
	int rev = (reverse ? -1 : 1);
	switch(ndims)
	{
		case 1:
			break;
		case 2:
			m = sqrtf(point[0]*point[0] + point[1]*point[1]);
			theta = atan2f(point[1], point[0]) + rev*rotation[0];
			point[0] = m*cosf(theta);
			point[1] = m*sinf(theta);
			break;
		case 3:
			r.x = rotation[0]; r.y = rotation[1]; r.z = rotation[2]; r.w = rotation[3];
			p.x = point[0]; p.y = point[1]; p.z = point[2];
			p.w = 0.0f;
			if(reverse)
				p = quaternion_mult(quaternion_conjugate(r), quaternion_mult(p, r));
			else
				p = quaternion_mult(r, quaternion_mult(p, quaternion_conjugate(r)));
			point[0] = p.x;
			point[1] = p.y;
			point[2] = p.z;
			break;
		default:
			printf("ERROR: Rotations in %d dimensional space are not supported!\n", ndims);
			break;
	}
}

DEVICE_FUNC
void drotate_point(const float* point, const int ndims, const float* rotation, int reverse, 
	const float* doutdpoint, float* doutdrotation)
{
	// int i, j, k;
	// float pp[3];
	// float base[3];
	// float rr[4];
	float4 r, p;
	float m, theta;
	int rev;
	switch(ndims)
	{
		case 1:
			break;
		case 2:
			rev = (reverse ? -1 : 1);
			m = sqrtf(point[0]*point[0] + point[1]*point[1]);
			theta = atan2f(point[1], point[0]) + rev*rotation[0];
			doutdrotation[0] = doutdpoint[0]*m*rev*-sinf(theta) + 
							   doutdpoint[1]*m*rev*cosf(theta);
		    break;			
		case 3:
			// for(k = 0; k < 3; ++k) base[k] = point[k];
			// for(k = 0; k < 4; ++k) rr[k] = rotation[k];
			// rotate_point(base, ndims, rotation, reverse);
			// for(i = 0; i < 4; ++i)
			// {
			// 	doutdrotation[i] = 0.0f;
			// 	for(k = 0; k < 3; ++k) pp[k] = point[k];
			// 	rr[i] += 1e-6;
			// 	rotate_point(pp, ndims, rr, reverse);
			// 	rr[i] -= 1e-6;
			// 	for(j = 0; j < 3; ++j)
			// 		doutdrotation[i] += doutdpoint[j]*(pp[j] - base[j])/1e-6;
			// }
			// printf(" %f,%f,%f,%f\n", doutdrotation[0], doutdrotation[1], doutdrotation[2],
			// 	doutdrotation[3]);
			r.x = rotation[0]; r.y = rotation[1]; r.z = rotation[2]; r.w = rotation[3];
			p.x = point[0]; p.y = point[1]; p.z = point[2];
			p.w = 0.0f;
			if(reverse)
			{
				doutdrotation[0] = doutdpoint[0]*(2*p.x*r.x + 2*p.y*r.y + 2*p.z*r.z) + 
				                   doutdpoint[1]*(2*p.x*r.y - 2*p.y*r.x + 2*p.z*r.w) + 
				                   doutdpoint[2]*(2*p.x*r.z - 2*p.y*r.w - 2*p.z*r.x);
				doutdrotation[1] = doutdpoint[0]*(-2*p.x*r.y + 2*p.y*r.x - 2*p.z*r.w) + 
				                   doutdpoint[1]*(2*p.x*r.x + 2*p.y*r.y + 2*p.z*r.z) + 
				                   doutdpoint[2]*(2*p.x*r.w + 2*p.y*r.z - 2*p.z*r.y);
				doutdrotation[2] = doutdpoint[0]*(-2*p.x*r.z + 2*p.y*r.w + 2*p.z*r.x) + 
				                   doutdpoint[1]*(-2*p.x*r.w - 2*p.y*r.z + 2*p.z*r.y) + 
				                   doutdpoint[2]*(2*p.x*r.x + 2*p.y*r.y + 2*p.z*r.z);
				doutdrotation[3] = doutdpoint[0]*(2*p.x*r.w + 2*p.y*r.z - 2*p.z*r.y) + 
				                   doutdpoint[1]*(-2*p.x*r.z + 2*p.y*r.w + 2*p.z*r.x) + 
				                   doutdpoint[2]*(2*p.x*r.y - 2*p.y*r.x + 2*p.z*r.w);
			}
			else
			{
				doutdrotation[0] = doutdpoint[0]*(2*p.x*r.x + 2*p.y*r.y + 2*p.z*r.z) + 
				                   doutdpoint[1]*(2*p.x*r.y - 2*p.y*r.x - 2*p.z*r.w) + 
				                   doutdpoint[2]*(2*p.x*r.z + 2*p.y*r.w - 2*p.z*r.x);
				doutdrotation[1] = doutdpoint[0]*(-2*p.x*r.y + 2*p.y*r.x + 2*p.z*r.w) + 
				                   doutdpoint[1]*(2*p.x*r.x + 2*p.y*r.y + 2*p.z*r.z) + 
				                   doutdpoint[2]*(-2*p.x*r.w + 2*p.y*r.z - 2*p.z*r.y);
				doutdrotation[2] = doutdpoint[0]*(-2*p.x*r.z - 2*p.y*r.w + 2*p.z*r.x) + 
				                   doutdpoint[1]*(2*p.x*r.w - 2*p.y*r.z + 2*p.z*r.y) + 
				                   doutdpoint[2]*(2*p.x*r.x + 2*p.y*r.y + 2*p.z*r.z);
				doutdrotation[3] = doutdpoint[0]*(2*p.x*r.w - 2*p.y*r.z + 2*p.z*r.y) + 
				                   doutdpoint[1]*(2*p.x*r.z + 2*p.y*r.w - 2*p.z*r.x) + 
				                   doutdpoint[2]*(-2*p.x*r.y + 2*p.y*r.x + 2*p.z*r.w);
			}
			// printf(">%f,%f,%f,%f\n", doutdrotation[0], doutdrotation[1], doutdrotation[2],
			// 	doutdrotation[3]);
			break;
		default:
			printf("ERROR: Rotations in %d dimensional space are not supported!\n", ndims);
			break;
	}
}

DEVICE_FUNC
void point_in_coordinate_frame(const float* point, const int ndims, 
	const float* translation, const float* rotation, float* out)
{
	int i;
	for(i = 0; i < ndims; ++i)
		out[i] = point[i] - translation[i];
	rotate_point(out, ndims, rotation, 1);
}

DEVICE_FUNC
float rec_nlinear_interp(const float* grid, const float* grid_dims, const int ndims, 
						const float* point01, int* lowidx, const int curdim, float* dpoint)
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
		float dpoint1[MAX_CARTESIAN_DIM];
		float* dp1 = (dpoint != NULL ? dpoint1 : NULL);
		float v1 = rec_nlinear_interp(grid, grid_dims, ndims, point01, lowidx, curdim + 1, dp1);
		lowidx[curdim] += 1;
		float dpoint2[MAX_CARTESIAN_DIM];
		float* dp2 = (dpoint != NULL ? dpoint2 : NULL);
		float v2 = rec_nlinear_interp(grid, grid_dims, ndims, point01, lowidx, curdim + 1, dp2);
		lowidx[curdim] -= 1;
		if(dpoint != NULL)
		{
			dpoint[curdim] = -v1 + v2;
			for(i = curdim + 1; i < ndims; ++i)
				dpoint[i] = (1 - point01[curdim])*dp1[i] + point01[curdim]*dp2[i];
		}
		return (1 - point01[curdim])*v1 + point01[curdim]*v2;
	}
}

DEVICE_FUNC
float nlinear_interp(const float* grid, const float* grid_dims, const int ndims, 
	const float cell_size, const float* point, float* dpoint)
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
	float ret = rec_nlinear_interp(grid, grid_dims, ndims, point01, lowidx, 0, dpoint);
	if(dpoint != NULL)
	{
		for(i = 0; i < ndims; ++i)
			dpoint[i] /= cell_size;
	}
	return ret;
}


/**
Function that that computes the partial values for the kernel cells for a given particle.
Given a particle index in a specific batch, this function loops through the given range
of particle indices, adding their contribution to each of the kernel cells around the
given particle. If any of the derivative variables (dqlocs, dlocs, ddata, or dweight) are
not NULL, then this funciton computes gradients instead of the output. In this case,
the variable out is assumed to contain the derivative. Inputs are:
	-qlocs: (batch_size X M X ndims) the cartesian coordinates of all the query locations.
	-locs: (batch_size X N X ndims) the cartesian coordinates of all the particles.
	-data: (batch_size X N X nchannels) the features associated with each particle.
	-neighbors: (batch_size X M X max_neighbors) a pre-computed list of neighbors for
				each particle. If there are fewer than max_neighbors for a given
				particle, the list is terminated in -1.
	-weight: (nkernels X nchannels X ncells) the kernel weights.
	-bias: (nkernels) the kernel biases.
	-batch_size: the size of the batch.
	-M: the number of query locations in each batch.
	-N: the number of particles in each batch.
	-nchannels: the number of features per particle.
	-ndims: the cardinality of the cartesian coordinate space.
	-max_neighbors: the maximum number of neighbors a given query location may have.
	-nkernels: the number of convolution kernels.
	-ncells: the number of cells in each kernel (this is the product of all the values
			 in kernel_size).
	-radius: the radius to use when doing neighbordhood lookups around a query point.
	-kernel_size: (ndims) the number of kernel cells in each dimension.
	-dilation: (ndims) the size of a signal kernel cell in each dimension.
	-kernel_fn: the id of the kernel function to use for W in the SPH equation.
	-dis_norm: divide the SPH values by the distance to the point.
	-out: (batch_size X M X nkernels) the partially computed output values for
		  each particle. If computing derivatives, contains the output derivative.
	-b: the batch index of the given query location.
	-n: the query location index of the given particle.
	-start: the particle index to start at when computing neighborhood lookups (inclusive).
	-end: the particle index to end at when computing neighborhood lookups (exclusive).
	-dqlocs: [Optional] (batch_size X M X ndims) if not NULL, then this function will 
			 compute the derivatives and place the one wrt to qlocs here.
	-dlocs: [Optional] (batch_size X N X ndims) if not NULL, then this function will 
	     	compute the derivatives and place the one wrt to locs here.
	-ddata: [Optional] (batch_size X N X nchannels) if not NULL, then this function will
			compute the derivative. This assumes that out is filled with the derivative
			of the output of this layer wrt some loss, and that this is initially
			filled with all 0s. The derivative wrt the data is stored here.
	-dweight [Optional] (nkernels X nchannels X ncells) similar to ddata, if not NULL
			 then the derivative wrt the weights is computed. It is expected that ddata
			 and dweight will either be both NULL (forward computation) or both not
			 NULL (backward computation).
	-bidirectional: if true, then contributions between two neighboring particles are added
					to both particles when the particle with the lowest index is encountered.
					When encountering the higher indexed particle, then that interaction is
					skipped. This can give almost a 2x speedup when true. This assumes qlocs == locs.
**/
DEVICE_FUNC
void compute_kernel_cells(
		const float* qlocs,
		const float* locs, 
		const float* data, 
		const float* neighbors,
		const float* weight, 
		const float* bias, 
		const int batch_size, 
		const int M,
		const int N, 
		const int nchannels, 
		const int ndims,
		const int max_neighbors,
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
		float* dqlocs,
		float* dlocs,
		float* ddata, 
		float* dweight)
{
	int idxs[MAX_CARTESIAN_DIM];
	float dkw[MAX_CARTESIAN_DIM];
	const float* r = qlocs + (b*M + n)*ndims;
	int backward = (dqlocs != NULL || dlocs != NULL || ddata != NULL || dweight != NULL);
	float* out_ptrn = out + b*nkernels*M + n*nkernels;
	const float* neighptr = neighbors + b*M*max_neighbors + n*max_neighbors;

	int j, jj;
	for(jj = 0; jj < max_neighbors && neighptr[jj] >= 0; ++jj)
	{
		j = neighptr[jj];
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
				const float* data_ptrj = data + b*nchannels*N + j*nchannels;
				float* ddata_ptrj = ddata + b*nchannels*N + j*nchannels;
				float kw = kernel_w(d, radius, kernel_fn);
				if(backward)
				{
					for(k = 0; k < ndims; ++k)
					{
						dkw[k] = kernel_dw(d, radius, kernel_fn)/d*
							(r[k] + (idxs[k] - ((int)kernel_size[k])/2)*dilation[k] - r2[k]);
					}
				}
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
						
						if(backward)
						{
							if(ddata != NULL)
							{
								atomicAdd(ddata_ptrj + ink, 
									(*(out_ptrn + outk))*weightnj*kw*norm);
							}
							if(dweight != NULL)
							{
								atomicAdd(dweight + outk*nchannels*ncells + 
										  ink*ncells + kernel_idx, 
									(*(out_ptrn + outk))*(*(data_ptrj + ink))*kw*norm);
							}
							if(dqlocs != NULL && d > 0)
							{
								for(k = 0; k < ndims; ++k)
								{
									atomicAdd(dqlocs + (b*M + n)*ndims + k,
										weightnj*(*(data_ptrj + ink))*norm*
										dkw[k]*(*(out_ptrn + outk)));
								}
							}
							if(dlocs != NULL && d > 0)
							{
								for(k = 0; k < ndims; ++k)
								{
									atomicAdd(dlocs + (b*N + j)*ndims + k,
										-weightnj*(*(data_ptrj + ink))*norm*
										dkw[k]*(*(out_ptrn + outk)));
								}
							}
						}
						else
						{
							float f1 = weightnj*(*(data_ptrj + ink))*kw*norm;
							atomicAdd(out_ptrn + outk, f1);
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
	-dlocs [Optional] (batch_size X N X ndims) if not NULL then the derivative wrt to
		   the locs is computed INSTEAD OF the partial sum. Assumes that out
		   contains the already computed sum.
	-dweight [Optional] (nkernels X ncells) if not NULL then the derivative wrt to
			 the weights is computed INSTEAD OF the partial sum. Assumes that out
			 contains the already computed sum.
	-dposes [Optional] (batch_size X M X pose_len) if not NULL then the derivative wrt to
			the poses is computed INSTEAD OF the partial sum. Assumes that out
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
		float* dlocs,
		float* dweight,
		float* dposes, 
		int* isdf_cache, 
		float* fsdf_cache)
{
	float r2[MAX_CARTESIAN_DIM];
	int backward = (dweight != NULL || dlocs != NULL || dposes != NULL);

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
			ndims, cell_size, r2, NULL)*scales[b*M + m];
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
		int smallest_m = -1;
		float s_dloc[MAX_CARTESIAN_DIM];
		float dloc[MAX_CARTESIAN_DIM];
		for(i = 0; i < ndims; ++i)
			s_dloc[i] = 0.0f;
		for(m = 0; m < M; ++m)
		{
			if(!isdf_cache[m])
				continue;
			
			float* dl = (backward ? dloc : NULL);
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
				ndims, cell_size, r2, dl)*scales[b*M + m];
			if(v < smallest)
			{
				smallest = v;
				smallest_m = m;
				if(backward)
				{
					for(i = 0; i < ndims; ++i)
						dloc[i] *= scales[b*M + m];
					rotate_point(dl, ndims, poses + b*M*pose_len + m*pose_len + ndims, 0);
					for(i = 0; i < ndims; ++i)
						s_dloc[i] = dloc[i];
				}
			}
		}
		if(backward)
		{
			if(dweight != NULL)
				atomicAdd(dweight + outk*ncells + kernel_idx, smallest*(*out_ptr));
			if(dlocs != NULL)
			{
				for(i = 0; i < ndims; ++i)
					atomicAdd(dlocs + (b*N + n)*ndims + i, 
						s_dloc[i]*(*out_ptr)*weight[outk*ncells + kernel_idx]);
			}
			if(dposes != NULL)
			{
				for(i = 0; i < ndims; ++i)
					atomicAdd(dposes + b*M*pose_len + smallest_m*pose_len + i, 
						-s_dloc[i]*(*out_ptr)*weight[outk*ncells + kernel_idx]);
				// dL/dq = dL/dv*dv/dq
				// dv/dq = dv/dr2*dr2/dq
				// Rotate s_dloc back so we can plug it in to the above.
				rotate_point(s_dloc, ndims, poses + b*M*pose_len + smallest_m*pose_len + ndims, 1);
				for(i = 0; i < ndims; ++i)
					dloc[i] = pt[i] - poses[b*M*pose_len + smallest_m*pose_len + i];
				drotate_point(dloc, ndims, poses + b*M*pose_len + smallest_m*pose_len + ndims, 1,
					s_dloc, r2);
				for(i = 0; i < pose_len - ndims; ++i)
					atomicAdd(dposes + b*M*pose_len + smallest_m*pose_len + ndims + i,
								r2[i]*(*out_ptr)*weight[outk*ncells + kernel_idx]);
			}
		}
		else
		{
			*out_ptr += weight[outk*ncells + kernel_idx]*smallest;
		}
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


/** Compute the neighbors for a given query location from a pre-computed hashgrid.
This function assumes that the size of an edge of a cell in the hashgrid is at least radius,
where radius is the maximum distance between any two locations to be considered neighbors. 
Furthermore it assumes that the locations have already been ordered according to the 
hashgrid, with all locations in the same grid cell contiguous in the locations array.
The arguments are:
	-qlocs (batch_size X M X dims) the cartesian coordinates of every query location.
	-locs (batch_size X N X dims) the cartesian coordinates of every particle.
	-cellStarts (batch_size X ncells) the index of the location in each given hashgrid cell
				with the smallest index.
	-cellEnds (batch_size X ncells) 1+ the maximum index of all locations in each hashgrid
			  cell.
	-batch_size: the size of the batch.
	-M: the number of query locations in each batch.
	-N: the number of particles in each batch.
	-ndims: the cardinality of the cartesian coordinate space.
	-ncells: the total number of hashgrid cells, equal to max_grid_dimension^dims.
	-low: (batch_size X dims) the cartesian coordinate of the lowest value for each hashgrid.
	-grid_dims: (batch_size X dims) the number of cells in each dimension for each hashgrid.
			    Note that the product of these values does not necessarily need to equal
			    ncells. grid_dims is used when indexing into cellStarts/cellEnds for each
			    specific hashgrid, but ncells is used when computing the stride from batch
			    to batch. cellStarts and cellEnds should have size batch_size*ncells, but
			    within each batch there may be unused values. ncells must always be at least
			    the product of grid_dims, but may be larger.
    -cellEdge: the size of one size of a cell in the hashgrids (all hashgrids must have the
    		   same cellEdge). Must be at least as large as radius.
	-radius2: the radius squared, which is the maximum distance between any two particles 
			  for them to be considered neighbors.
	-collisions: (batch_size X M X max_collisions) the list of location indices that are
				 neighbors for every location. Not all locations will have max_collisions
				 neighbors, so the list for each location is terminated with a -1.
	-b: the batch to compute neighbors for.
	-n: the query location to compute neighbors for.
**/
DEVICE_FUNC
void compute_collisions(
	const float* qlocs,
	const float* locs,
	const float* cellStarts,
	const float* cellEnds,
	const int batch_size,
	const int M,
	const int N,
	const int ndims,
	const int ncells,
	const float* low,
	const float* grid_dims,
	const float cellEdge,
	const float radius2,
	float* collisions,
	const int max_collisions,
	const int include_self,
	const int b,
	const int n)
{
	int ncollisions = 0;
	int grid_coord[MAX_CARTESIAN_DIM];
	int offset[MAX_CARTESIAN_DIM];
	int k, i;
	const float* r = qlocs + (b*M + n)*ndims;
	for(k = 0; k < ndims; ++k)
	{
		grid_coord[k] = loc2grid(r[k], low[b*ndims + k], cellEdge);
		offset[k] = -1;
	}
	while(offset[ndims-1] <= 1 && ncollisions < max_collisions)
	{
		int inbounds = 1;
		int cellID = 0;
		for(k = 0; k < ndims && inbounds; ++k)
		{
			int coord = grid_coord[k] + offset[k];
			if(coord < 0 || coord >= grid_dims[b*ndims + k])
				inbounds = 0;
			else
				cellID += partial_grid_hash(coord, grid_dims + b*ndims, k, ndims);
		}

		if(inbounds)
		{
			for(i = cellStarts[b*ncells + cellID]; i < cellEnds[b*ncells + cellID] &&
					ncollisions < max_collisions; ++i)
			{
				float d = 0.0f;
				for(k = 0; k < ndims; ++k)
				{
					float nr = r[k] - locs[b*N*ndims + i*ndims + k];
					d += nr*nr;
				}
				if(d < radius2 && (d > 0 || include_self))
				{
					collisions[b*M*max_collisions + n*max_collisions + ncollisions] = i;
					++ncollisions;
				}
			}
		}

		++offset[0];
		for(k = 0; k < ndims - 1 && offset[k] > 1; ++k)
		{
			offset[k] = -1;
			++offset[k+1];
		}
	}

	if(ncollisions < max_collisions)
		collisions[b*M*max_collisions + n*max_collisions + ncollisions] = -1;
}


/**
Function that that computes the projection of the particles onto the image plane. Projects
each particle as a circular gaussian. Sums the contributions from every particle at each
pixel. Note that unlike all other functions in this file, this function only works with
particles in 3D. Inputs are:
	-locs: (batch_size X N X 3) the cartesian coordinates of all the particles in camera
		   space.
	-batch_size: the size of the batch.
	-N: the number of particles in each batch.
	-camera_fl: focal length of the camera in pixels.
	-width: width of the camera image.
	-height: height of the camera image.
	-filter_std: standard deviation of the gaussian to project.
	-filter_scale: amount to scale the gaussian values by.
	-depth_mask: [Optional] (batch_size X height X width) mask used to block projections of
			     particles. For any pixel a particle projects onto, if that particle is 
			     behind the value in the mask, the particle's contribution to that pixel is
			     ignored.
    -b: the batch index of the particle being projected.
	-n: the index of the particle being projected.
	-out: (batch_size X height X width) the output with the projection of particle n in batch
		  b added to it. If derivatives are being computed (dlocs is not NULL), then this is
		  assumed to contain the derivative of out wrt the loss.
	-dlocs: [Optional] (batch_size X N X 3) if not NULL, then this function assumes that out 
	     	contains the derivative of some value wrt to the output, and so this function
	     	will compute the derivatives instead of the normal output and place the one wrt 
	     	to locs here.
**/
DEVICE_FUNC
void compute_particle_projection(
	const float* locs,
	const int batch_size,
	const int N,
	const float camera_fl,
	const int width,
	const int height,
	const float filter_std,
	const float filter_scale,
	const float* depth_mask,
	const int n,
	const int b,
	float* out,
	float* dlocs
)
{
	int backward = (dlocs != NULL);
	const float* rr = locs + (b*N + n)*3;
	float4 r; r.x = rr[0]; r.y = rr[1]; r.z = rr[2]; r.w = 0;
	// Can't contribute if it's behind the camera.
	if(r.z <= 0) return;
	float4 proj;
	proj.x = r.x*camera_fl/r.z + width/2;
	proj.y = r.y*camera_fl/r.z + height/2;
	float i, j;
	int s = ceilf(filter_std*2);
	float s2 = s*s;
	float f = filter_scale/(filter_std*sqrtf(2*M_PI));
	float std2 = filter_std*filter_std;
	for(i = MAX(proj.x - s, 0); i < width && i < proj.x + s + 1; i += 1)
	{
		for(j = MAX(proj.y - s, 0); j < height && j < proj.y + s + 1; j += 1)
		{
			int ii = i;
			int jj = j;
			float depth_val = *(depth_mask + b*width*height + jj*width + ii);
			if(depth_val > 0.0f && depth_val < r.z)
				continue;
			float xi = ii + 0.5f;
			float yj = jj + 0.5f;
			float d2 = (xi - proj.x)*(xi - proj.x) + (yj - proj.y)*(yj - proj.y);
			if(d2 > s2)
				continue;
			float v = f*expf(-d2/(2.0f*std2));
			if(!backward)
			{
				atomicAdd(out + b*width*height + jj*width + ii, v);
			}
			else
			{
				// dL/dlocs = dL/dout*dout/dproj*dproj/dlocs
				// dL/dout -> Given in dlocs
				// dout/dproj -> Derivative of v wrt proj
				//      = [(proj.x - xi)*v/std2, 
				//         (proj.y - yj)*v/std2]
				// dproj/locs -> Derivative of proj wrt locs
				//      = [[camera_fl/r.z, 0], 
				//		   [0, camera_fl/r.z], 
				//		   [-r.x*camera_fl/(r.z*r.z), -r.y*camera_fl/(r.z*r.z)]]
				float dLdout = *(out + b*width*height + jj*width + ii);
				// We can multiply out dL/dout*dout/dproj*dproj/dtran and compute it
				// all at once to get dL/dtran.
				atomicAdd(dlocs + (b*N + n)*3 + 0, 
					dLdout*(xi - proj.x)*v/std2*camera_fl/r.z);
				atomicAdd(dlocs + (b*N + n)*3 + 1, 
					dLdout*(yj - proj.y)*v/std2*camera_fl/r.z);
				atomicAdd(dlocs + (b*N + n)*3 + 2, 
					dLdout*v/std2*camera_fl/(r.z*r.z)*
					((xi - proj.x)*-r.x + (yj - proj.y)*-r.y));
			}
		}	
	}
}


/**
Function that that computes the projection of an image onto a set of particles. For each
particle, it determines its image coordinates, then uses bilinear inerpolation on the
image features to compute features for that particle.
Note that unlike all other functions in this file, this function only works with
particles in 3D. Inputs are:
	-locs: (batch_size X N X 3) the cartesian coordinates of all the particles in camera
		   space.
	-image: (batch_size X channels X height X width) the image to project onto the
		    particles.
	-batch_size: the size of the batch.
	-N: the number of particles in each batch.
	-camera_fl: focal length of the camera in pixels.
	-width: width of the camera image.
	-height: height of the camera image.
	-channels: the number of channels in the image.
	-depth_mask: [Optional] (batch_size X height X width) mask used to block projections of
			     particles. For any pixel a particle projects onto, if that particle is 
			     behind the value in the mask, the particle gets no feature values.
    -b: the batch index of the particle being projected.
	-n: the index of the particle being projected.
	-out: (batch_size X N X channels) the output with the projection of particle n in batch
		  b added to it. If derivatives are being computed (dlocs is not NULL), then this is
		  assumed to contain the derivative of out wrt the loss.
	-dlocs: [Optional] (batch_size X N X 3) if not NULL, then this function assumes that out 
	     	contains the derivative of some value wrt to the output, and so this function
	     	will compute the derivatives instead of the normal output and place the one wrt 
	     	to locs here.
 	-dimage: [Optional] (batch_size X channels X height X width) similar to dlocs, if not
 			 NULL then the derivative of some value wrt to the image is computed and
 			 placed here.
**/
DEVICE_FUNC
void compute_image_projection(
	const float* locs,
	const float* image,
	const int batch_size,
	const int N,
	const float camera_fl,
	const int width,
	const int height,
	const int channels,
	const float* depth_mask,
	const int n,
	const int b,
	float* out,
	float* dlocs,
	float* dimage
)
{
	int backward = (dlocs != NULL || dimage != NULL);
	const float* rr = locs + (b*N + n)*3;
	float4 r; r.x = rr[0]; r.y = rr[1]; r.z = rr[2]; r.w = 0;
	// Can't contribute if it's behind the camera.
	if(r.z <= 0) return;
	float4 proj;
	proj.x = r.x*camera_fl/r.z + width/2;
	proj.y = r.y*camera_fl/r.z + height/2;
	// Make sure the particle is in bounds.
	if(proj.x <= 0.5 || proj.x >= width - 0.5 || proj.y <= 0.5 || proj.y >= height - 0.5)
		return;
	int ii = proj.x;
	int jj = proj.y;
	// Check the depth mask.
	float depth_val = *(depth_mask + b*width*height + jj*width + ii);
	if(depth_val > 0.0f && depth_val < r.z)
		return;
	int c;
	int lowi, highi, lowj, highj;
	float vll, vlh, vhl, vhh, v, di, dj;
	const float* image_ptr;
	for(c = 0; c < channels; ++c)
	{
		// float v = nlinear_interp(image + b*channels*width*height + c*width*height, 
		// 	dims, 2, 1, rr, dl);
		// I chose not to use nlinear_interp here even though it does exactly what I
		// want because computing derivatives wrt the input grid would require an
		// exponential in the dimensionality sized buffer. Since this function only
		// works for 3D points, it is okay to do it this way.
		lowi = proj.x - 0.5;
		highi = proj.x + 0.5;
		lowj = proj.y - 0.5;
		highj = proj.y + 0.5;
		di = proj.x - 0.5 - lowi;
		dj = proj.y - 0.5 - lowj;
		image_ptr = image + b*channels*width*height + c*width*height;
		vll = image_ptr[lowj*width + lowi];
		vlh = image_ptr[highj*width + lowi];
		vhl = image_ptr[lowj*width + highi];
		vhh = image_ptr[highj*width + highi];
		v = vll*(1 - di)*(1 - dj) +
			vlh*(1 - di)*dj +
			vhl*di*(1 - dj) +
			vhh*di*dj;
		
		
		if(!backward)
		{
			atomicAdd(out + b*N*channels + n*channels + c, v);
		}
		else
		{
			// dL/dlocs = dL/dout*dout/dproj*dproj/dlocs
			// dL/dout -> Given in dlocs
			// dout/dproj -> Derivative of v wrt proj
			//      Given by nlinear_interp
			// dproj/locs -> Derivative of proj wrt locs
			//      = [[camera_fl/r.z, 0], 
			//		   [0, camera_fl/r.z], 
			//		   [-r.x*camera_fl/(r.z*r.z), -r.y*camera_fl/(r.z*r.z)]]
			float dLdout = *(out + b*N*channels + n*channels + c);
			float doutpx = -vll*(1 - dj) + -vlh*dj + vhl*(1 - dj) + vhh*dj;
			float doutpy = -vll*(1 - di) + vlh*(1 - di) + -vhl*di + vhh*di;
			// We can multiply out dL/dout*dout/dproj*dproj/dtran and compute it
			// all at once to get dL/dtran.
			atomicAdd(dlocs + (b*N + n)*3 + 0, 
				dLdout*camera_fl/r.z*doutpx);
			atomicAdd(dlocs + (b*N + n)*3 + 1, 
				dLdout*camera_fl/r.z*doutpy);
			atomicAdd(dlocs + (b*N + n)*3 + 2, 
				dLdout*-r.x*camera_fl/(r.z*r.z)*doutpx + dLdout*-r.y*camera_fl/(r.z*r.z)*doutpy);
			// Derivative wrt image.
			atomicAdd(dimage + b*channels*width*height + c*width*height + lowj*width + lowi, dLdout*(1 - di)*(1 - dj));
			atomicAdd(dimage + b*channels*width*height + c*width*height + highj*width + lowi, dLdout*(1 - di)*dj);
			atomicAdd(dimage + b*channels*width*height + c*width*height + lowj*width + highi, dLdout*di*(1 - dj));
			atomicAdd(dimage + b*channels*width*height + c*width*height + highj*width + highi, dLdout*di*dj);
		}
	}
}


#ifdef __cplusplus
}
#endif
#endif
