



int spnc_fgrid_forward_cuda(THCudaTensor *locs, THCudaTensor *value_grid, THCudaTensor *output);

int spnc_fgrid_backward_cuda(THCudaTensor *locs, THCudaTensor *grad_output, THCudaTensor *grad_input);

