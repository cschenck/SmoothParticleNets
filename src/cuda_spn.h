size_t spnc_get_shared_mem_size(int device);

int spnc_convsp_forward(THCudaTensor* locs_t, THCudaTensor* data_t, 
    THCudaTensor* weight_t, THCudaTensor* bias_t, float radius, 
    THCudaTensor* kernel_size_t, THCudaTensor* dilation_t, int dis_norm, int kernel_fn, 
    THCudaTensor* out_t, size_t nshared_device_mem);

int spnc_convsp_backward(THCudaTensor* locs_t, THCudaTensor* data_t, 
    THCudaTensor* weight_t, THCudaTensor* bias_t, float radius, 
    THCudaTensor* kernel_size_t, THCudaTensor* dilation_t, int dis_norm, int kernel_fn, 
    THCudaTensor* out_t, THCudaTensor* ddata_t, THCudaTensor* dweight_t, 
    size_t nshared_device_mem);

int spnc_convsdf_forward(THCudaTensor* locs_t, THCudaTensor* idxs_t, THCudaTensor* poses_t, 
    THCudaTensor* scales_t, THCudaTensor* sdfs_t, THCudaTensor* sdf_offsets_t,
    THCudaTensor* sdf_shapes_t, THCudaTensor* weight_t, THCudaTensor* bias_t, 
    THCudaTensor* kernel_size_t, THCudaTensor* dilation_t, float max_distance,
    THCudaTensor* out_t);

int spnc_convsdf_backward(THCudaTensor* locs_t, THCudaTensor* idxs_t, THCudaTensor* poses_t, 
    THCudaTensor* scales_t, THCudaTensor* sdfs_t, THCudaTensor* sdf_offsets_t,
    THCudaTensor* sdf_shapes_t, THCudaTensor* weight_t, THCudaTensor* bias_t, 
    THCudaTensor* kernel_size_t, THCudaTensor* dilation_t, float max_distance,
    THCudaTensor* out_t, THCudaTensor* dweight_t);