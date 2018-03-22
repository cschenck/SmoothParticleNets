


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

int spnc_compute_collisions(void* locs_t, 
                           void* data_t, 
                           void* lower_bounds_t,
                           void* grid_dims_t,
                           void* cellIDs_t,
                           void* idxs_t,
                           void* cellStarts_t,
                           void* cellEnds_t,
                           void* collisions_t,
                           void* buffer_t,
                           const float cellEdge,
                           const float radius);

size_t spnc_get_radixsort_buffer_size(void);