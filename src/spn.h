

int spn_max_cartesian_dim(void);


int spn_convsp_forward(THFloatTensor* locs_t, THFloatTensor* data_t, THFloatTensor* density_t, 
    THFloatTensor* weight_t, THFloatTensor* bias_t,float radius, THFloatTensor* kernel_size_t, 
    THFloatTensor* dilation_t, THFloatTensor* out_t);

int spn_convsp_backward(THFloatTensor* locs_t, THFloatTensor* data_t, THFloatTensor* density_t, 
    THFloatTensor* weight_t, THFloatTensor* bias_t,float radius, THFloatTensor* kernel_size_t, 
    THFloatTensor* dilation_t, THFloatTensor* out_t, THFloatTensor* ddata_t,
    THFloatTensor* dweight_t);