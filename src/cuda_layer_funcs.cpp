
#include <stdio.h>

#include <torch/torch.h>
#include <THC/THC.h>


#include "gpu_kernels.h"


size_t spnc_get_shared_mem_size(int device)
{
    return GetSharedMemPerBlock(device);
}


int spnc_convsp_forward(const at::Tensor qlocs_t, const at::Tensor locs_t, 
    const at::Tensor data_t, const at::Tensor neighbors_t,
    const at::Tensor weight_t, const at::Tensor bias_t, const float radius, 
    const at::Tensor kernel_size_t, const at::Tensor dilation_t, const int dis_norm, 
    const int kernel_fn, at::Tensor out_t, const size_t nshared_device_mem)
{
    const float* qlocs = (const float*)qlocs_t.data_ptr();
    const float* locs = (const float*)locs_t.data_ptr();
    const float* data = (const float*)data_t.data_ptr();
    const float* neighbors = (const float*)neighbors_t.data_ptr();
    const float* weight = (const float*)weight_t.data_ptr();
    const float* bias = (const float*)bias_t.data_ptr();
    const int batch_size = locs_t.sizes()[0];
    const int M = qlocs_t.sizes()[1];
    const int N = locs_t.sizes()[1];
    const int nchannels = data_t.sizes()[2];
    const int ndims = locs_t.sizes()[2];
    const int max_neighbors = neighbors_t.sizes()[2];
    const int nkernels = weight_t.sizes()[0];
    const int ncells = weight_t.sizes()[2];
    const float* kernel_size = (const float*)kernel_size_t.data_ptr();
    const float* dilation = (const float*)dilation_t.data_ptr();
    float* out = (float*)out_t.data_ptr();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    return cuda_convsp(qlocs, locs, data, neighbors, weight, bias, batch_size, M, N, nchannels, ndims,
        max_neighbors, nkernels, ncells, radius, kernel_size, dilation, dis_norm, kernel_fn, 
        out, NULL, NULL, NULL, NULL, stream, nshared_device_mem);

}

int spnc_convsp_backward(const at::Tensor qlocs_t, const at::Tensor locs_t, 
    const at::Tensor data_t, const at::Tensor neighbors_t,
    const at::Tensor weight_t, const at::Tensor bias_t, const float radius, 
    const at::Tensor kernel_size_t, const at::Tensor dilation_t, const int dis_norm, 
    const int kernel_fn, at::Tensor out_t, at::Tensor dqlocs_t, at::Tensor dlocs_t, 
    at::Tensor ddata_t, at::Tensor dweight_t, const size_t nshared_device_mem)
{
    const float* qlocs = (const float*)qlocs_t.data_ptr();
    const float* locs = (const float*)locs_t.data_ptr();
    const float* data = (const float*)data_t.data_ptr();
    const float* neighbors = (const float*)neighbors_t.data_ptr();
    const float* weight = (const float*)weight_t.data_ptr();
    const float* bias = (const float*)bias_t.data_ptr();
    float* dqlocs = (float*)dqlocs_t.data_ptr();
    float* dlocs = (float*)dlocs_t.data_ptr();
    float* ddata = (float*)ddata_t.data_ptr();
    float* dweight = (float*)dweight_t.data_ptr();
    const int batch_size = locs_t.sizes()[0];
    const int M = qlocs_t.sizes()[1];
    const int N = locs_t.sizes()[1];
    const int nchannels = data_t.sizes()[2];
    const int ndims = locs_t.sizes()[2];
    const int max_neighbors = neighbors_t.sizes()[2];
    const int nkernels = weight_t.sizes()[0];
    const int ncells = weight_t.sizes()[2];
    const float* kernel_size = (const float*)kernel_size_t.data_ptr();
    const float* dilation = (const float*)dilation_t.data_ptr();
    float* out = (float*)out_t.data_ptr();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    return cuda_convsp(qlocs, locs, data, neighbors, weight, bias, batch_size, M, N, nchannels, ndims,
        max_neighbors, nkernels, ncells, radius, kernel_size, dilation, dis_norm, kernel_fn, 
        out, dqlocs, dlocs, ddata, dweight, stream, nshared_device_mem);
}

int spnc_convsdf_forward(const at::Tensor locs_t, const at::Tensor idxs_t, 
    const at::Tensor poses_t, const at::Tensor scales_t, const at::Tensor sdfs_t, 
    const at::Tensor sdf_offsets_t, const at::Tensor sdf_shapes_t, 
    const at::Tensor weight_t, const at::Tensor bias_t, 
    const at::Tensor kernel_size_t, const at::Tensor dilation_t, 
    const float max_distance, at::Tensor out_t)
{
    const float* locs = (const float*)locs_t.data_ptr();
    const float* idxs = (const float*)idxs_t.data_ptr();
    const float* poses = (const float*)poses_t.data_ptr();
    const float* scales = (const float*)scales_t.data_ptr();
    const float* sdfs = (const float*)sdfs_t.data_ptr();
    const float* sdf_offsets = (const float*)sdf_offsets_t.data_ptr();
    const float* sdf_shapes = (const float*)sdf_shapes_t.data_ptr();
    const float* weight = (const float*)weight_t.data_ptr();
    const float* bias = (const float*)bias_t.data_ptr(); 
    const int batch_size = locs_t.sizes()[0];
    const int N = locs_t.sizes()[1];
    const int ndims = locs_t.sizes()[2];
    const int M = idxs_t.sizes()[1];
    const int pose_len = poses_t.sizes()[2];
    const int nkernels = weight_t.sizes()[0];
    const int ncells = weight_t.sizes()[1];
    const float* kernel_size = (const float*)kernel_size_t.data_ptr();
    const float* dilation = (const float*)dilation_t.data_ptr();
    float* out = (float*)out_t.data_ptr();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    return cuda_convsdf(locs, batch_size, N, ndims, idxs, poses, scales, M, pose_len, sdfs, 
        sdf_offsets, sdf_shapes, weight, bias, nkernels, ncells, kernel_size, dilation, 
        max_distance, out, NULL, NULL, NULL, stream);
}

int spnc_convsdf_backward(const at::Tensor locs_t, const at::Tensor idxs_t, 
    const at::Tensor poses_t, const at::Tensor scales_t, const at::Tensor sdfs_t, 
    const at::Tensor sdf_offsets_t, const at::Tensor sdf_shapes_t, 
    const at::Tensor weight_t, const at::Tensor bias_t, 
    const at::Tensor kernel_size_t, const at::Tensor dilation_t, 
    const float max_distance, at::Tensor out_t, at::Tensor dlocs_t, 
    at::Tensor dweight_t, at::Tensor dposes_t)
{
    const float* locs = (const float*)locs_t.data_ptr();
    const float* idxs = (const float*)idxs_t.data_ptr();
    const float* poses = (const float*)poses_t.data_ptr();
    const float* scales = (const float*)scales_t.data_ptr();
    const float* sdfs = (const float*)sdfs_t.data_ptr();
    const float* sdf_offsets = (const float*)sdf_offsets_t.data_ptr();
    const float* sdf_shapes = (const float*)sdf_shapes_t.data_ptr();
    const float* weight = (const float*)weight_t.data_ptr();
    const float* bias = (const float*)bias_t.data_ptr(); 
    float* dlocs = (float*)dlocs_t.data_ptr(); 
    float* dweight = (float*)dweight_t.data_ptr(); 
    float* dposes = (float*)dposes_t.data_ptr(); 
    const int batch_size = locs_t.sizes()[0];
    const int N = locs_t.sizes()[1];
    const int ndims = locs_t.sizes()[2];
    const int M = idxs_t.sizes()[1];
    const int pose_len = poses_t.sizes()[2];
    const int nkernels = weight_t.sizes()[0];
    const int ncells = weight_t.sizes()[1];
    const float* kernel_size = (const float*)kernel_size_t.data_ptr();
    const float* dilation = (const float*)dilation_t.data_ptr();
    float* out = (float*)out_t.data_ptr();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Computing dposes will cause lots of thread clashes, so only compute it if absolutely necessary.
    if(dposes_t.sizes()[0] != batch_size)
        dposes = NULL;

    return cuda_convsdf(locs, batch_size, N, ndims, idxs, poses, scales, M, pose_len, sdfs, 
        sdf_offsets, sdf_shapes, weight, bias, nkernels, ncells, kernel_size, dilation, 
        max_distance, out, dlocs, dweight, dposes, stream);
}

int spnc_hashgrid_order(at::Tensor locs_t, 
                           at::Tensor lower_bounds_t,
                           at::Tensor grid_dims_t,
                           at::Tensor cellIDs_t,
                           at::Tensor idxs_t,
                           at::Tensor buffer_t,
                           const float cellEdge)
{
    float* locs = (float*)locs_t.data_ptr();
    float* low = (float*)lower_bounds_t.data_ptr();
    float* grid_dims = (float*)grid_dims_t.data_ptr();
    float* cellIDs = (float*)cellIDs_t.data_ptr();
    float* idxs = (float*)idxs_t.data_ptr();
    float* buffer = (float*)buffer_t.data_ptr();
    const int batch_size = locs_t.sizes()[0];
    const int N = locs_t.sizes()[1];
    const int ndims = locs_t.sizes()[2];
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    return cuda_hashgrid_order(locs, low, grid_dims, cellIDs, idxs,
        buffer, batch_size, N, ndims, cellEdge, stream);
}

int spnc_compute_collisions(at::Tensor qlocs_t, 
                           at::Tensor locs_t, 
                           at::Tensor lower_bounds_t,
                           at::Tensor grid_dims_t,
                           at::Tensor cellIDs_t,
                           at::Tensor cellStarts_t,
                           at::Tensor cellEnds_t,
                           at::Tensor collisions_t,
                           const float cellEdge,
                           const float radius,
                           const int include_self)
{
    float* qlocs = (float*)qlocs_t.data_ptr();
    float* locs = (float*)locs_t.data_ptr();
    float* low = (float*)lower_bounds_t.data_ptr();
    float* grid_dims = (float*)grid_dims_t.data_ptr();
    float* cellIDs = (float*)cellIDs_t.data_ptr();
    float* cellStarts = (float*)cellStarts_t.data_ptr();
    float* cellEnds = (float*)cellEnds_t.data_ptr();
    float* collisions = (float*)collisions_t.data_ptr();
    const int batch_size = locs_t.sizes()[0];
    const int M = qlocs_t.sizes()[1];
    const int N = locs_t.sizes()[1];
    const int ndims = locs_t.sizes()[2];
    const int max_collisions = collisions_t.sizes()[2];
    const int ncells = cellStarts_t.sizes()[1];
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    return cuda_compute_collisions(qlocs, locs, low, grid_dims, cellIDs, cellStarts,
        cellEnds, collisions, batch_size, M, N, ndims, max_collisions, 
        ncells, cellEdge, radius, include_self, stream);
}

int spnc_reorder_data(at::Tensor locs_t, 
                         at::Tensor data_t, 
                         at::Tensor idxs_t,
                         at::Tensor nlocs_t,
                         at::Tensor ndata_t,
                         const int reverse)
{
    float* locs = (float*)locs_t.data_ptr();
    float* data = NULL;
    float* idxs = (float*)idxs_t.data_ptr();
    float* nlocs = (float*)nlocs_t.data_ptr();
    float* ndata = NULL;
    const int batch_size = locs_t.sizes()[0];
    const int N = locs_t.sizes()[1];
    const int ndims = locs_t.sizes()[2];
    int nchannels = 0;
    if(data_t.defined())
    {
        nchannels = data_t.sizes()[2];
        data = (float*)data_t.data_ptr();
        ndata = (float*)ndata_t.data_ptr();
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    return cuda_reorder_data(
            locs,
            data,
            idxs,
            nlocs,
            ndata,
            batch_size,
            N,
            ndims,
            nchannels,
            reverse,
            stream);
}

size_t spnc_get_radixsort_buffer_size(void)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    return get_radixsort_buffer_size(stream);
}

int spnc_particleprojection_forward(at::Tensor locs_t,
                                   const float camera_fl,
                                   const float filter_std,
                                   const float filter_scale,
                                   at::Tensor depth_mask_t,
                                   at::Tensor out_t)
{
    float* locs = (float*)locs_t.data_ptr();
    float* depth_mask = (float*)depth_mask_t.data_ptr();
    float* out = (float*)out_t.data_ptr();
    const int batch_size = locs_t.sizes()[0];
    const int N = locs_t.sizes()[1];
    const int width = depth_mask_t.sizes()[2];
    const int height = depth_mask_t.sizes()[1];
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    return cuda_particleprojection(locs,
                                   camera_fl,
                                   filter_std,
                                   filter_scale,
                                   depth_mask,
                                   batch_size,
                                   N,
                                   width,
                                   height,
                                   out,
                                   NULL,
                                   stream);
}

int spnc_particleprojection_backward(at::Tensor locs_t,
                                   const float camera_fl,
                                   const float filter_std,
                                   const float filter_scale,
                                   at::Tensor depth_mask_t,
                                   at::Tensor out_t,
                                   at::Tensor dlocs_t)
{
    float* locs = (float*)locs_t.data_ptr();
    float* depth_mask = (float*)depth_mask_t.data_ptr();
    float* out = (float*)out_t.data_ptr();
    float* dlocs = (float*)dlocs_t.data_ptr();
    const int batch_size = locs_t.sizes()[0];
    const int N = locs_t.sizes()[1];
    const int width = depth_mask_t.sizes()[2];
    const int height = depth_mask_t.sizes()[1];
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    return cuda_particleprojection(locs,
                                   camera_fl,
                                   filter_std,
                                   filter_scale,
                                   depth_mask,
                                   batch_size,
                                   N,
                                   width,
                                   height,
                                   out,
                                   dlocs,
                                   stream);
}


int spnc_imageprojection_forward(at::Tensor locs_t,
                                   at::Tensor image_t,
                                   const float camera_fl,
                                   at::Tensor depth_mask_t,
                                   at::Tensor out_t)
{
    float* locs = (float*)locs_t.data_ptr();
    float* image = (float*)image_t.data_ptr();
    float* depth_mask = (float*)depth_mask_t.data_ptr();
    float* out = (float*)out_t.data_ptr();
    const int batch_size = locs_t.sizes()[0];
    const int N = locs_t.sizes()[1];
    const int width = image_t.sizes()[3];
    const int height = image_t.sizes()[2];
    const int channels = image_t.sizes()[1];
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    return cuda_imageprojection(locs,
                                   image,
                                   camera_fl,
                                   depth_mask,
                                   batch_size,
                                   N,
                                   width,
                                   height,
                                   channels,
                                   out,
                                   NULL,
                                   NULL,
                                   stream);
}

int spnc_imageprojection_backward(at::Tensor locs_t,
                                   at::Tensor image_t,
                                   const float camera_fl,
                                   at::Tensor depth_mask_t,
                                   at::Tensor out_t,
                                   at::Tensor dlocs_t,
                                   at::Tensor dimage_t)
{
    float* locs = (float*)locs_t.data_ptr();
    float* image = (float*)image_t.data_ptr();
    float* depth_mask = (float*)depth_mask_t.data_ptr();
    float* out = (float*)out_t.data_ptr();
    float* dlocs = (float*)dlocs_t.data_ptr();
    float* dimage = (float*)dimage_t.data_ptr();
    const int batch_size = locs_t.sizes()[0];
    const int N = locs_t.sizes()[1];
    const int width = image_t.sizes()[3];
    const int height = image_t.sizes()[2];
    const int channels = image_t.sizes()[1];
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    return cuda_imageprojection(locs,
                                   image,
                                   camera_fl,
                                   depth_mask,
                                   batch_size,
                                   N,
                                   width,
                                   height,
                                   channels,
                                   out,
                                   dlocs,
                                   dimage,
                                   stream);
}


 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("spnc_convsp_forward", &spnc_convsp_forward, "ConvSP Forward");
    m.def("spnc_convsp_backward", &spnc_convsp_backward, "ConvSP Backward");
    m.def("spnc_convsdf_forward", &spnc_convsdf_forward, "ConvSDF Forward");
    m.def("spnc_convsdf_backward", &spnc_convsdf_backward, "ConvSDF Backward");
    m.def("spnc_hashgrid_order", &spnc_hashgrid_order, "Hashgrid Order");
    m.def("spnc_compute_collisions", &spnc_compute_collisions, "Compute Collisions");
    m.def("spnc_reorder_data", &spnc_reorder_data, "Reorder Data");
    m.def("spnc_particleprojection_backward", &spnc_particleprojection_backward, "Particle Projection Backward");
    m.def("spnc_particleprojection_forward", &spnc_particleprojection_forward, "Particle Projection Forward");
    m.def("spnc_imageprojection_backward", &spnc_imageprojection_backward, "Image Projection Backward");
    m.def("spnc_imageprojection_forward", &spnc_imageprojection_forward, "Image Projection Forward");
    m.def("spnc_get_shared_mem_size", &spnc_get_shared_mem_size, "Maximum size of device shared memory");
    m.def("spnc_get_radixsort_buffer_size", &spnc_get_radixsort_buffer_size, "Buffer size for radix sort");
}
