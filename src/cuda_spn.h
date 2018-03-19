size_t spnc_get_shared_mem_size(int device);

int spnc_convsp_forward(const THCudaTensor* locs_t, const THCudaTensor* data_t, 
    const THCudaTensor* weight_t, const THCudaTensor* bias_t, const float radius, 
    const THCudaTensor* kernel_size_t, const THCudaTensor* dilation_t, const int dis_norm, 
    const int kernel_fn, THCudaTensor* out_t, const size_t nshared_device_mem);

int spnc_convsp_backward(const THCudaTensor* locs_t, const THCudaTensor* data_t, 
    const THCudaTensor* weight_t, const THCudaTensor* bias_t, const float radius, 
    const THCudaTensor* kernel_size_t, const THCudaTensor* dilation_t, const int dis_norm, 
    const int kernel_fn, THCudaTensor* out_t, THCudaTensor* ddata_t, THCudaTensor* dweight_t, 
    const size_t nshared_device_mem);

int spnc_convsdf_forward(const THCudaTensor* locs_t, const THCudaTensor* idxs_t, 
    const THCudaTensor* poses_t, const THCudaTensor* scales_t, const THCudaTensor* sdfs_t, 
    const THCudaTensor* sdf_offsets_t, const THCudaTensor* sdf_shapes_t, 
    const THCudaTensor* weight_t, const THCudaTensor* bias_t, 
    const THCudaTensor* kernel_size_t, const THCudaTensor* dilation_t, 
    const float max_distance, THCudaTensor* out_t);

int spnc_convsdf_backward(const THCudaTensor* locs_t, const THCudaTensor* idxs_t, 
    const THCudaTensor* poses_t, const THCudaTensor* scales_t, const THCudaTensor* sdfs_t, 
    const THCudaTensor* sdf_offsets_t, const THCudaTensor* sdf_shapes_t, 
    const THCudaTensor* weight_t, const THCudaTensor* bias_t, 
    const THCudaTensor* kernel_size_t, const THCudaTensor* dilation_t, 
    const float max_distance, THCudaTensor* out_t, THCudaTensor* dweight_t);