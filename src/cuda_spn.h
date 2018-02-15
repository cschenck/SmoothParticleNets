int spnc_convsp_forward(THCudaTensor* locs_t, THCudaTensor* data_t, THCudaTensor* density_t,
    THCudaTensor* cellIdxs_t, THCudaTensor* originalIndex_t, THCudaTensor* cellStart_t,
    THCudaTensor* cellEnd_t, THCudaTensor* gridShape_t,
    THCudaTensor* weight_t, THCudaTensor* bias_t, float radius, 
    THCudaTensor* kernel_size_t, THCudaTensor* dilation_t, THCudaTensor* out_t);

int spnc_convsp_backward(THCudaTensor* locs_t, THCudaTensor* data_t, THCudaTensor* density_t,
    THCudaTensor* cellIdxs_t, THCudaTensor* originalIndex_t, THCudaTensor* cellStart_t,
    THCudaTensor* cellEnd_t, THCudaTensor* gridShape_t, 
    THCudaTensor* weight_t, THCudaTensor* bias_t, float radius, 
    THCudaTensor* kernel_size_t, THCudaTensor* dilation_t, THCudaTensor* out_t,
    THCudaTensor* ddata_t, THCudaTensor* dweight_t);

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