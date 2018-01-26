int spnc_convsp_forward(void* locs_t, void* data_t, void* density_t, 
    void* weight_t, void* bias_t,float radius, void* kernel_size_t, 
    void* dilation_t, void* out_t);

int spnc_convsp_backward(void* locs_t, void* data_t, void* density_t, 
    void* weight_t, void* bias_t,float radius, void* kernel_size_t, 
    void* dilation_t, void* out_t, void* ddata_t,
    void* dweight_t);