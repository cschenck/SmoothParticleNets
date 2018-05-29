

int spn_max_cartesian_dim(void);


int spn_convsp_forward(const THFloatTensor* qlocs_t, const THFloatTensor* locs_t, const THFloatTensor* data_t, 
    const THFloatTensor* neighbors_t, const THFloatTensor* weight_t, 
    const THFloatTensor* bias_t, const float radius, 
    const THFloatTensor* kernel_size_t, const THFloatTensor* dilation_t, 
    const int dis_norm, const int kernel_fn, THFloatTensor* out_t);

int spn_convsp_backward(const THFloatTensor* qlocs_t, const THFloatTensor* locs_t, const THFloatTensor* data_t, 
    const THFloatTensor* neighbors_t, const THFloatTensor* weight_t, 
    const THFloatTensor* bias_t, const float radius, 
    const THFloatTensor* kernel_size_t, const THFloatTensor* dilation_t, 
    const int dis_norm, const int kernel_fn, THFloatTensor* out_t, 
    THFloatTensor* dqlocs_t, THFloatTensor* dlocs_t,
    THFloatTensor* ddata_t, THFloatTensor* dweight_t);

int spn_convsdf_forward(const THFloatTensor* locs_t, const THFloatTensor* idxs_t, 
    const THFloatTensor* poses_t, const THFloatTensor* scales_t, 
    const THFloatTensor* sdfs_t, const THFloatTensor* sdf_offsets_t,
    const THFloatTensor* sdf_shapes_t, const THFloatTensor* weight_t, 
    const THFloatTensor* bias_t, const THFloatTensor* kernel_size_t, 
    const THFloatTensor* dilation_t, const float max_distance,
    THFloatTensor* out_t);

int spn_convsdf_backward(const THFloatTensor* locs_t, const THFloatTensor* idxs_t, 
    const THFloatTensor* poses_t, const THFloatTensor* scales_t, const THFloatTensor* sdfs_t, 
    const THFloatTensor* sdf_offsets_t, const THFloatTensor* sdf_shapes_t, 
    const THFloatTensor* weight_t, const THFloatTensor* bias_t, 
    const THFloatTensor* kernel_size_t, const THFloatTensor* dilation_t, 
    const float max_distance, THFloatTensor* out_t, THFloatTensor* dlocs_t, 
    THFloatTensor* dweight_t, THFloatTensor* dposes_t);

int spn_hashgrid_order(THFloatTensor* locs_t, 
                           THFloatTensor* lower_bounds_t,
                           THFloatTensor* grid_dims_t,
                           THFloatTensor* cellIDs_t,
                           THFloatTensor* idxs_t,
                           const float cellEdge);

int spn_compute_collisions(THFloatTensor* qlocs_t,
                           THFloatTensor* locs_t, 
                           THFloatTensor* lower_bounds_t,
                           THFloatTensor* grid_dims_t,
                           THFloatTensor* cellIDs_t,
                           THFloatTensor* cellStarts_t,
                           THFloatTensor* cellEnds_t,
                           THFloatTensor* collisions_t,
                           const float cellEdge,
                           const float radius,
                           const int include_self);

int spn_reorder_data(THFloatTensor* locs_t, 
                         THFloatTensor* data_t, 
                         THFloatTensor* idxs_t,
                         THFloatTensor* nlocs_t,
                         THFloatTensor* ndata_t,
                         const int reverse);

int spn_particleprojection_forward(THFloatTensor* locs_t,
                                   const float camera_fl,
                                   const float filter_std,
                                   const float filter_scale,
                                   THFloatTensor* depth_mask_t,
                                   THFloatTensor* out_t);

int spn_particleprojection_backward(THFloatTensor* locs_t,
                                   const float camera_fl,
                                   const float filter_std,
                                   const float filter_scale,
                                   THFloatTensor* depth_mask_t,
                                   THFloatTensor* out_t,
                                   THFloatTensor* dlocs_t);