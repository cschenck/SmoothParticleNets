
#include <TH/TH.h>
#include <THC/THC.h>

#include "gpu_kernels.h"

extern THCState *state;


int spnc_particles2grid_forward_cuda(THCudaTensor *locs, THCudaTensor *data, THCudaTensor *density, 
	THCudaTensor *grid, float grid_lowerx, float grid_lowery, float grid_lowerz, float grid_stepsx,
	float grid_stepsy, float grid_stepsz, float radius)
{
	locs = THCudaTensor_newContiguous(state, locs);
	data = THCudaTensor_newContiguous(state, data);
	density = THCudaTensor_newContiguous(state, density);
	grid = THCudaTensor_newContiguous(state, grid);

	float* points = THCudaTensor_data(state, locs);
	float* pdata = THCudaTensor_data(state, data);
	float* pdensity = THCudaTensor_data(state, density);
	int batch_size = locs->size[0];
	int nparticles = locs->size[1];
	int data_dims = data->size[2];
	int grid_dimsx = grid->size[1];
	int grid_dimsy = grid->size[2];
	int grid_dimsz = grid->size[3];
	float* pgrid = THCudaTensor_data(state, grid);
	cudaStream_t stream = THCState_getCurrentStream(state);

	return cuda_particles2grid(points, pdata, pdensity, nparticles, batch_size, data_dims,
		grid_lowerx, grid_lowery, grid_lowerz, grid_dimsx, grid_dimsy, grid_dimsz,
		grid_stepsx, grid_stepsy, grid_stepsz, pgrid, radius, stream);
}
