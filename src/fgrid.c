

#include <TH/TH.h>
#include <THC/THC.h>

#include "gpu_kernels.h"

extern THCState *state;


int spnc_fgrid_forward_cuda(THCudaTensor *locs, THCudaTensor *value_grid, THCudaTensor *output)
{
	locs = THCudaTensor_newContiguous(state, locs);
	value_grid = THCudaTensor_newContiguous(state, value_grid);
	output = THCudaTensor_newContiguous(state, output);
	
	float* plocs = THCudaTensor_data(state, locs);
	float* pdata = THCudaTensor_data(state, value_grid);
	int batch_size = locs->size[0];
	int nlocs = locs->size[1];
	int dlocs = locs->size[2];
	int dim_sizes[MAX_TENSOR_DIM];
	int i;
	for(i = 0; i + 1 < value_grid->nDimension; ++i)
		dim_sizes[i] = value_grid->size[i + 1];
	int ddata = value_grid->nDimension - dlocs - 1;
	const int* data_dims = dim_sizes + ddata;
	float* out = THCudaTensor_data(state, output);
	cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_assign_from_locs(plocs, pdata, batch_size, nlocs, dlocs, dim_sizes, ddata, data_dims, out, stream);
}


int spnc_fgrid_backward_cuda(THCudaTensor *locs, THCudaTensor *grad_output, THCudaTensor *grad_input)
{
	locs = THCudaTensor_newContiguous(state, locs);
	grad_output = THCudaTensor_newContiguous(state, grad_output);
	grad_input = THCudaTensor_newContiguous(state, grad_input);
	
	float* plocs = THCudaTensor_data(state, locs);
	float* pdata = THCudaTensor_data(state, grad_output);
	int batch_size = locs->size[0];
	int nlocs = locs->size[1];
	int dlocs = locs->size[2];
	int dim_sizes[MAX_TENSOR_DIM];
	int i;
	for(i = 0; i + 1 < grad_input->nDimension; ++i)
		dim_sizes[i] = grad_input->size[i + 1];
	int ddata = grad_input->nDimension - dlocs - 1;
	const int* data_dims = dim_sizes + ddata;
	float* out = THCudaTensor_data(state, grad_input);
	cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_add_to_locs(plocs, pdata, batch_size, nlocs, dlocs, dim_sizes, ddata, data_dims, out, stream);
}