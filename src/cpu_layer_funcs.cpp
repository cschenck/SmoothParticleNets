
#include <string.h>

#include <torch/torch.h>

#ifdef CUDA
#undef CUDA
#endif
#include "common_funcs.h"
#include "constants.h"


int cpu_convsp(const float* qlocs, const float* locs, const float* data, const float* neighbors, 
    const float* weight, const float* bias, 
    const int batch_size, const int M, const int N, const int nchannels, const int ndims, 
    const int max_neighbors,
    const int nkernels, const int ncells, const float radius, const float* kernel_size, 
    const float* dilation, const int dis_norm, const int kernel_fn, float* out, 
    float* dqlocs, float* dlocs, float* ddata, float* dweight);

int cpu_convsdf(const float* locs, const int batch_size, const int N, const int ndims, 
    const float* idxs, const float* poses, const float* scales, const int M, 
    const int pose_len, const float* sdfs, const float* sdf_offsets, const float* sdf_shapes, 
    const float* weight, const float* bias, const int nkernels, const int ncells, 
    const float* kernel_size, const float* dilation, const float max_distance, 
    float* out, float* dlocs, float* dweight, float* dposes);


int spn_max_cartesian_dim(void)
{
    return MAX_CARTESIAN_DIM;
}

int spn_convsp_forward(const at::Tensor qlocs_t, const at::Tensor locs_t, 
    const at::Tensor data_t, 
    const at::Tensor neighbors_t, const at::Tensor weight_t, 
    const at::Tensor bias_t, const float radius, 
    const at::Tensor kernel_size_t, const at::Tensor dilation_t, 
    const int dis_norm, const int kernel_fn, at::Tensor out_t)
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

    return cpu_convsp(qlocs, locs, data, neighbors, weight, bias, batch_size, M, 
        N, nchannels, ndims, max_neighbors,
        nkernels, ncells, radius, kernel_size, dilation, dis_norm, kernel_fn, out, NULL, 
        NULL, NULL, NULL);
}

int spn_convsp_backward(const at::Tensor qlocs_t, const at::Tensor locs_t, 
    const at::Tensor data_t, 
    const at::Tensor neighbors_t, const at::Tensor weight_t, 
    const at::Tensor bias_t, const float radius, 
    const at::Tensor kernel_size_t, const at::Tensor dilation_t, 
    const int dis_norm, const int kernel_fn, at::Tensor out_t, 
    at::Tensor dqlocs_t, at::Tensor dlocs_t,
    at::Tensor ddata_t, at::Tensor dweight_t)
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

    return cpu_convsp(qlocs, locs, data, neighbors, weight, bias, batch_size, M, 
        N, nchannels, ndims, max_neighbors,
        nkernels, ncells, radius, kernel_size, dilation, dis_norm, kernel_fn, out, 
        dqlocs, dlocs, ddata, dweight);
}

int cpu_convsp(const float* qlocs, const float* locs, const float* data, const float* neighbors, 
    const float* weight, const float* bias, 
    const int batch_size, const int M, const int N, const int nchannels, const int ndims, 
    const int max_neighbors,
    const int nkernels, const int ncells, const float radius, const float* kernel_size, 
    const float* dilation, const int dis_norm, const int kernel_fn, float* out, 
    float* dqlocs, float* dlocs, float* ddata, float* dweight)
{
    int b, n;
    for(b = 0; b < batch_size; ++b)
    {
        for(n = 0; n < M; ++n)
        {
            compute_kernel_cells(qlocs, locs, data, neighbors, weight, bias, batch_size, M, N, 
                nchannels, ndims, max_neighbors, nkernels, ncells, radius, kernel_size,
                dilation, dis_norm, kernel_fn, out, b, n, dqlocs, dlocs, ddata, dweight);
        }
    }
    return 1;
}




int spn_convsdf_forward(const at::Tensor locs_t, const at::Tensor idxs_t, 
    const at::Tensor poses_t, const at::Tensor scales_t, 
    const at::Tensor sdfs_t, const at::Tensor sdf_offsets_t,
    const at::Tensor sdf_shapes_t, const at::Tensor weight_t, 
    const at::Tensor bias_t, const at::Tensor kernel_size_t, 
    const at::Tensor dilation_t, const float max_distance,
    at::Tensor out_t)
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

    return cpu_convsdf(locs, batch_size, N, ndims, idxs, poses, scales, M, pose_len, sdfs, 
        sdf_offsets, sdf_shapes, weight, bias, nkernels, ncells, kernel_size, dilation, 
        max_distance, out, NULL, NULL, NULL);
}

int spn_convsdf_backward(const at::Tensor locs_t, const at::Tensor idxs_t, 
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

    // Computing dposes will cause lots of thread clashes, so only compute it if absolutely necessary.
    if(dposes_t.sizes()[0] != batch_size)
        dposes = NULL;

    return cpu_convsdf(locs, batch_size, N, ndims, idxs, poses, scales, M, pose_len, sdfs, 
        sdf_offsets, sdf_shapes, weight, bias, nkernels, ncells, kernel_size, dilation, 
        max_distance, out, dlocs, dweight, dposes);
}

int cpu_convsdf(const float* locs, const int batch_size, const int N, const int ndims, 
    const float* idxs, const float* poses, const float* scales, const int M, 
    const int pose_len, const float* sdfs, const float* sdf_offsets, const float* sdf_shapes, 
    const float* weight, const float* bias, const int nkernels, const int ncells, 
    const float* kernel_size, const float* dilation, const float max_distance, 
    float* out, float* dlocs, float* dweight, float* dposes)
{
    int* isdf_cache = (int*)malloc(sizeof(int)*M);
    float* fsdf_cache = (float*)malloc(sizeof(float)*M);
    int b, n, outk;
    for(b = 0; b < batch_size; ++b)
    {
        for(n = 0; n < N; ++n)
        {
            for(outk = 0; outk < nkernels; ++outk)
            {
                compute_sdf_kernel_cells(locs, batch_size, N, ndims, idxs, poses, 
                    scales, M, pose_len, sdfs, sdf_offsets, sdf_shapes, weight, bias, 
                    nkernels, ncells, kernel_size, dilation, max_distance, out, b, n,
                    outk, dlocs, dweight, dposes, isdf_cache, fsdf_cache);
            }
        }
    }
    free(isdf_cache);
    free(fsdf_cache);
    return 1;
}


int spn_hashgrid_order(at::Tensor locs_t, 
                           at::Tensor lower_bounds_t,
                           at::Tensor grid_dims_t,
                           at::Tensor cellIDs_t,
                           at::Tensor idxs_t,
                           const float cellEdge)
{
    float* locs = (float*)locs_t.data_ptr();
    float* low = (float*)lower_bounds_t.data_ptr();
    float* grid_dims = (float*)grid_dims_t.data_ptr();
    float* cellIDs = (float*)cellIDs_t.data_ptr();
    float* idxs = (float*)idxs_t.data_ptr();
    const int batch_size = locs_t.sizes()[0];
    const int N = locs_t.sizes()[1];
    const int ndims = locs_t.sizes()[2];

    int b, i, j, d;

    // Mark each particle with it's cell ID.
    for(b = 0; b < batch_size; ++b)
    {
        for(i = 0; i < N; ++i)
        {
            int hash = 0;
            for(d = 0; d < ndims; ++d)
                hash += partial_grid_hash(
                    loc2grid(locs[b*N*ndims + i*ndims + d], low[b*ndims + d], cellEdge), 
                    grid_dims + b*ndims, d, ndims);
            cellIDs[b*N + i] = hash;
            idxs[b*N + i] = i;
        }
    }

    // Sort the particles by cell ID.
    for(b = 0; b < batch_size; ++b)
    {
        // Going to use slow selection sort because it's easy to implement.
        for(i = 0; i < N; ++i)
        {
            int minID = cellIDs[b*N + i];
            int minidx = i;
            for(j = i + 1; j < N; ++j)
            {
                // Rather than swapping in place now, do it later for efficiency.
                if(cellIDs[b*N + j] < minID)
                {
                    minID = cellIDs[b*N + j];
                    minidx = j;
                }
            }

            if(minidx != i)
            {
                swapf(cellIDs + b*N + i, cellIDs + b*N + minidx, 1);
                swapf(idxs + b*N + i, idxs + b*N + minidx, 1);
            }
        }
    }

    return 1;
}

int spn_compute_collisions(at::Tensor qlocs_t,
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

    int b, i;
    // Create the cell start and end lists.
    for(b = 0; b < batch_size; ++b)
    {
        for(i = 0; i < N; ++i)
        {
            int c = cellIDs[b*N + i];
            if (i == 0)
            {
                cellStarts[b*ncells + c] = i;
            }
            else
            {
                int p = cellIDs[b*N + i-1];

                if (c != p)
                {
                    cellStarts[b*ncells + c] = i;
                    cellEnds[b*ncells + p] = i;
                }
            }
            
            if (i == N-1)
            {
                cellEnds[b*ncells + c] = i+1;
            }
        }
    }

    // Make collision lists.
    for(b = 0; b < batch_size; ++b)
    {
        for(i = 0; i < M; ++i)
        {
            compute_collisions(
                qlocs,
                locs,
                cellStarts,
                cellEnds,
                batch_size,
                M,
                N,
                ndims,
                ncells,
                low,
                grid_dims,
                cellEdge,
                radius*radius,
                collisions,
                max_collisions,
                include_self,
                b,
                i);
        }
    }

    return 1;
}

int spn_reorder_data(at::Tensor locs_t, 
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
    if(data_t.defined() > 0)
    {
        nchannels = data_t.sizes()[2];
        data = (float*)data_t.data_ptr();
        ndata = (float*)ndata_t.data_ptr();
    }

    int b, i, d;
    for(b = 0; b < batch_size; ++b)
    {
        for(i = 0; i < N; ++i)
        {
            int nn = i;
            int on = idxs[b*N + i];
            if(reverse)
            {
                nn = idxs[b*N + i];
                on = i%N;
            }
            for(d = 0; d < ndims; ++d)
                nlocs[b*N*ndims + nn*ndims + d] = locs[b*N*ndims + on*ndims + d];
            if(data != NULL)
            {
                for(d = 0; d < nchannels; ++d)
                    ndata[b*N*nchannels + nn*nchannels + d] = 
                                    data[b*N*nchannels + on*nchannels + d];
            }
        }
    }

    return 1;
}


int spn_particleprojection_backward(at::Tensor locs_t,
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
    float* dlocs = NULL;
    if(dlocs_t.defined() > 0)
        dlocs = (float*)dlocs_t.data_ptr();
    const int batch_size = locs_t.sizes()[0];
    const int N = locs_t.sizes()[1];
    const int width = out_t.sizes()[2];
    const int height = out_t.sizes()[1];
    int b, i;
    for(b = 0; b < batch_size; ++b)
    {
        for(i = 0; i < N; ++i)
        {
            compute_particle_projection(
                locs,
                batch_size,
                N,
                camera_fl,
                width,
                height,
                filter_std,
                filter_scale,
                depth_mask,
                i,
                b,
                out,
                dlocs
            );
        }
    }

    return 1;
}

int spn_particleprojection_forward(at::Tensor locs_t,
                                   const float camera_fl,
                                   const float filter_std,
                                   const float filter_scale,
                                   at::Tensor depth_mask_t,
                                   at::Tensor out_t)
{
    return spn_particleprojection_backward(locs_t, 
                                          camera_fl,
                                          filter_std,
                                          filter_scale,
                                          depth_mask_t,
                                          out_t,
                                          at::Tensor());
}


int spn_imageprojection_backward(at::Tensor locs_t,
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
    float* dlocs = NULL;
    if(dlocs_t.defined() > 0)
        dlocs = (float*)dlocs_t.data_ptr();
    float* dimage = NULL;
    if(dimage_t.defined() > 0)
        dimage = (float*)dimage_t.data_ptr();
    const int batch_size = locs_t.sizes()[0];
    const int N = locs_t.sizes()[1];
    const int channels = image_t.sizes()[1];
    const int width = image_t.sizes()[3];
    const int height = image_t.sizes()[2];
    int b, i;
    for(b = 0; b < batch_size; ++b)
    {
        for(i = 0; i < N; ++i)
        {
            compute_image_projection(
                locs,
                image,
                batch_size,
                N,
                camera_fl,
                width,
                height,
                channels,
                depth_mask,
                i,
                b,
                out,
                dlocs,
                dimage
            );
        }
    }

    return 1;
}

int spn_imageprojection_forward(at::Tensor locs_t,
                                 at::Tensor image_t,
                                   const float camera_fl,
                                   at::Tensor depth_mask_t,
                                   at::Tensor out_t)
{
    return spn_imageprojection_backward(locs_t, 
                                        image_t,
                                          camera_fl,
                                          depth_mask_t,
                                          out_t,
                                          at::Tensor(),
                                          at::Tensor());
}


 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("spn_convsp_forward", &spn_convsp_forward, "ConvSP Forward");
    m.def("spn_convsp_backward", &spn_convsp_backward, "ConvSP Backward");
    m.def("spn_convsdf_forward", &spn_convsdf_forward, "ConvSDF Forward");
    m.def("spn_convsdf_backward", &spn_convsdf_backward, "ConvSDF Backward");
    m.def("spn_hashgrid_order", &spn_hashgrid_order, "Hashgrid Order");
    m.def("spn_compute_collisions", &spn_compute_collisions, "Compute Collisions");
    m.def("spn_reorder_data", &spn_reorder_data, "Reorder Data");
    m.def("spn_particleprojection_backward", &spn_particleprojection_backward, "Particle Projection Backward");
    m.def("spn_particleprojection_forward", &spn_particleprojection_forward, "Particle Projection Forward");
    m.def("spn_imageprojection_backward", &spn_imageprojection_backward, "Image Projection Backward");
    m.def("spn_imageprojection_forward", &spn_imageprojection_forward, "Image Projection Forward");
    m.def("spn_max_cartesian_dim", &spn_max_cartesian_dim, "Maximum Number of Dimensions");
}
