size_t spnc_get_shared_mem_size(int device);

int spnc_convsp_forward(const THCudaTensor* qlocs_t, const THCudaTensor* locs_t, const THCudaTensor* data_t, 
    const THCudaTensor* neighbors_t,
    const THCudaTensor* weight_t, const THCudaTensor* bias_t, const float radius, 
    const THCudaTensor* kernel_size_t, const THCudaTensor* dilation_t, const int dis_norm, 
    const int kernel_fn, THCudaTensor* out_t, const size_t nshared_device_mem);

int spnc_convsp_backward(const THCudaTensor* qlocs_t, const THCudaTensor* locs_t, const THCudaTensor* data_t, 
    const THCudaTensor* neighbors_t,
    const THCudaTensor* weight_t, const THCudaTensor* bias_t, const float radius, 
    const THCudaTensor* kernel_size_t, const THCudaTensor* dilation_t, const int dis_norm, 
    const int kernel_fn, THCudaTensor* out_t, THCudaTensor* dqlocs_t, THCudaTensor* dlocs_t, 
    THCudaTensor* ddata_t, THCudaTensor* dweight_t, const size_t nshared_device_mem);

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
    const float max_distance, THCudaTensor* out_t, THCudaTensor* dlocs_t, 
    THCudaTensor* dweight_t, THCudaTensor* dposes_t);

int spnc_hashgrid_order(THCudaTensor* locs_t, 
                           THCudaTensor* lower_bounds_t,
                           THCudaTensor* grid_dims_t,
                           THCudaTensor* cellIDs_t,
                           THCudaTensor* idxs_t,
                           THCudaTensor* buffer_t,
                           const float cellEdge);

int spnc_compute_collisions(THCudaTensor* qlocs_t, 
                           THCudaTensor* locs_t, 
                           THCudaTensor* lower_bounds_t,
                           THCudaTensor* grid_dims_t,
                           THCudaTensor* cellIDs_t,
                           THCudaTensor* cellStarts_t,
                           THCudaTensor* cellEnds_t,
                           THCudaTensor* collisions_t,
                           const float cellEdge,
                           const float radius,
                           const int include_self);

int spnc_reorder_data(THCudaTensor* locs_t, 
                         THCudaTensor* data_t, 
                         THCudaTensor* idxs_t,
                         THCudaTensor* nlocs_t,
                         THCudaTensor* ndata_t,
                         const int reverse);

size_t spnc_get_radixsort_buffer_size(void);

int spnc_particleprojection_forward(THCudaTensor* locs_t,
                                   const float camera_fl,
                                   const float filter_std,
                                   const float filter_scale,
                                   THCudaTensor* depth_mask_t,
                                   THCudaTensor* out_t);

int spnc_particleprojection_backward(THCudaTensor* locs_t,
                                   const float camera_fl,
                                   const float filter_std,
                                   const float filter_scale,
                                   THCudaTensor* depth_mask_t,
                                   THCudaTensor* out_t,
                                   THCudaTensor* dlocs_t);

int spnc_imageprojection_forward(THCudaTensor* locs_t,
                                   THCudaTensor* image_t,
                                   const float camera_fl,
                                   THCudaTensor* depth_mask_t,
                                   THCudaTensor* out_t);

int spnc_imageprojection_backward(THCudaTensor* locs_t,
                                   THCudaTensor* image_t,
                                   const float camera_fl,
                                   THCudaTensor* depth_mask_t,
                                   THCudaTensor* out_t,
                                   THCudaTensor* dlocs_t,
                                   THCudaTensor* dimage_t);