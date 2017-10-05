



int spnc_fgrid_forward_cuda(THCudaTensor *locs, THCudaTensor *value_grid, THCudaTensor *output);

int spnc_fgrid_backward_cuda(THCudaTensor *locs, THCudaTensor *grad_output, THCudaTensor *grad_input);

int spnc_particles2grid_forward_cuda(THCudaTensor *locs, THCudaTensor *data, THCudaTensor *density, 
	THCudaTensor *grid, float grid_lowerx, float grid_lowery, float grid_lowerz, float grid_stepsx,
	float grid_stepsy, float grid_stepsz, float radius);

int spnc_grid2particles_forward_cuda(THCudaTensor *grid, float grid_lowerx, float grid_lowery, 
	float grid_lowerz, float grid_stepsx, float grid_stepsy, float grid_stepsz, 
	THCudaTensor *locs, THCudaTensor *data);
