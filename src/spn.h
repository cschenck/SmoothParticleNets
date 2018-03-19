

int spn_max_cartesian_dim(void);


int spn_convsp_forward(const THFloatTensor* locs_t, const THFloatTensor* data_t, 
    const THFloatTensor* weight_t, const THFloatTensor* bias_t, const float radius, 
    const THFloatTensor* kernel_size_t, const THFloatTensor* dilation_t, 
    const int dis_norm, const int kernel_fn, THFloatTensor* out_t);

int spn_convsp_backward(const THFloatTensor* locs_t, const THFloatTensor* data_t, 
    const THFloatTensor* weight_t, const THFloatTensor* bias_t, const float radius, 
    const THFloatTensor* kernel_size_t, const THFloatTensor* dilation_t, 
    const int dis_norm, const int kernel_fn, THFloatTensor* out_t, 
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
    const float max_distance, THFloatTensor* out_t, THFloatTensor* dweight_t);