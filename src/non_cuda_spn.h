


size_t spnc_get_shared_mem_size(int device);

int spnc_convsp_forward(const void* locs_t, const void* data_t, 
    const void* weight_t, const void* bias_t, const float radius, 
    const void* kernel_size_t, const void* dilation_t, const int dis_norm, 
    const int kernel_fn, void* out_t, const size_t nshared_device_mem);

int spnc_convsp_backward(const void* locs_t, const void* data_t, 
    const void* weight_t, const void* bias_t, const float radius, 
    const void* kernel_size_t, const void* dilation_t, const int dis_norm, 
    const int kernel_fn, void* out_t, void* ddata_t, void* dweight_t, 
    const size_t nshared_device_mem);

int spnc_convsdf_forward(const void* locs_t, const void* idxs_t, 
    const void* poses_t, const void* scales_t, const void* sdfs_t, 
    const void* sdf_offsets_t, const void* sdf_shapes_t, 
    const void* weight_t, const void* bias_t, 
    const void* kernel_size_t, const void* dilation_t, 
    const float max_distance, void* out_t);

int spnc_convsdf_backward(const void* locs_t, const void* idxs_t, 
    const void* poses_t, const void* scales_t, const void* sdfs_t, 
    const void* sdf_offsets_t, const void* sdf_shapes_t, 
    const void* weight_t, const void* bias_t, 
    const void* kernel_size_t, const void* dilation_t, 
    const float max_distance, void* out_t, void* dweight_t);