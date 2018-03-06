


size_t spnc_get_shared_mem_size(int device);

int spnc_convsp_forward(void* locs_t, void* data_t, void* density_t, 
    void* weight_t, void* bias_t,float radius, void* kernel_size_t, 
    void* dilation_t, int kernel_fn, void* out_t, size_t nshared_device_mem);

int spnc_convsp_backward(void* locs_t, void* data_t, void* density_t, 
    void* weight_t, void* bias_t,float radius, void* kernel_size_t, 
    void* dilation_t, int kernel_fn, void* out_t, void* ddata_t,
    void* dweight_t, size_t nshared_device_mem);

int spnc_convsdf_forward(void** locs_t, void** idxs_t, void** poses_t, 
    void** scales_t, void** sdfs_t, void** sdf_offsets_t,
    void** sdf_shapes_t, void** weight_t, void** bias_t, 
    void** kernel_size_t, void** dilation_t, float max_distance,
    void** out_t);

int spnc_convsdf_backward(void** locs_t, void** idxs_t, void** poses_t, 
    void** scales_t, void** sdfs_t, void** sdf_offsets_t,
    void** sdf_shapes_t, void** weight_t, void** bias_t, 
    void** kernel_size_t, void** dilation_t, float max_distance,
    void** out_t, void** dweight_t);