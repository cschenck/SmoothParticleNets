int spnc_convsp_forward(THCudaTensor* locs_t, THCudaTensor* data_t, THCudaTensor* density_t, 
    THCudaTensor* weight_t, THCudaTensor* bias_t, float radius, 
    THCudaTensor* kernel_size_t, THCudaTensor* dilation_t, THCudaTensor* out_t);

int spnc_convsp_backward(THCudaTensor* locs_t, THCudaTensor* data_t, THCudaTensor* density_t, 
    THCudaTensor* weight_t, THCudaTensor* bias_t, float radius, 
    THCudaTensor* kernel_size_t, THCudaTensor* dilation_t, THCudaTensor* out_t,
    THCudaTensor* ddata_t, THCudaTensor* dweight_t);