
#include <TH/TH.h>
#include <THC/THC.h>

#include "gpu_kernels.h"

extern THCState *state;


int spnc_grid2particles_forward_cuda(THCudaTensor *grid, float grid_lowerx, float grid_lowery, 
	float grid_lowerz, float grid_stepsx, float grid_stepsy, float grid_stepsz, 
	THCudaTensor *locs, THCudaTensor *data)
{
	grid = THCudaTensor_newContiguous(state, grid);
	locs = THCudaTensor_newContiguous(state, locs);
	data = THCudaTensor_newContiguous(state, data);

	float* points = THCudaTensor_data(state, locs);
	float* pdata = THCudaTensor_data(state, data);
	int batch_size = locs->size[0];
	int nparticles = locs->size[1];
	int data_dims = data->size[2];
	int grid_dimsx = grid->size[1];
	int grid_dimsy = grid->size[2];
	int grid_dimsz = grid->size[3];
	float* pgrid = THCudaTensor_data(state, grid);
	cudaStream_t stream = THCState_getCurrentStream(state);

	return cuda_grid2particles(pgrid, batch_size,
		grid_lowerx, grid_lowery, grid_lowerz, grid_dimsx, grid_dimsy, grid_dimsz,
		grid_stepsx, grid_stepsy, grid_stepsz, data_dims, points, nparticles, pdata, stream);
}
