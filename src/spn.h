

int spn_max_cartesian_dim(void);


int spn_convsp_forward(THFloatTensor* locs_t, THFloatTensor* data_t, THFloatTensor* density_t, 
    THFloatTensor* neighborlist_t, THFloatTensor* weight_t, THFloatTensor* bias_t,float radius, 
    THFloatTensor* kernel_size_t, 
    THFloatTensor* dilation_t, THFloatTensor* out_t);

int spn_convsp_backward(THFloatTensor* locs_t, THFloatTensor* data_t, THFloatTensor* density_t, 
    THFloatTensor* neighborlist_t, THFloatTensor* weight_t, THFloatTensor* bias_t,
    float radius, THFloatTensor* kernel_size_t, 
    THFloatTensor* dilation_t, THFloatTensor* out_t, THFloatTensor* ddata_t,
    THFloatTensor* dweight_t);

int spn_convsdf_forward(THFloatTensor* locs_t, THFloatTensor* idxs_t, THFloatTensor* poses_t, 
    THFloatTensor* scales_t, THFloatTensor* sdfs_t, THFloatTensor* sdf_offsets_t,
    THFloatTensor* sdf_shapes_t, THFloatTensor* weight_t, THFloatTensor* bias_t, 
    THFloatTensor* kernel_size_t, THFloatTensor* dilation_t, float max_distance,
    THFloatTensor* out_t);

int spn_convsdf_backward(THFloatTensor* locs_t, THFloatTensor* idxs_t, THFloatTensor* poses_t, 
    THFloatTensor* scales_t, THFloatTensor* sdfs_t, THFloatTensor* sdf_offsets_t,
    THFloatTensor* sdf_shapes_t, THFloatTensor* weight_t, THFloatTensor* bias_t, 
    THFloatTensor* kernel_size_t, THFloatTensor* dilation_t, float max_distance,
    THFloatTensor* out_t, THFloatTensor* dweight_t);