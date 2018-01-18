#include <TH/TH.h>
#include <THC/THC.h>

#include "gpu_kernels.h"

extern THCState *state;


/* Particles2Grid Layer */

int spnc_particles2grid_forward_cuda(THCudaTensor *locs, THCudaTensor *data, THCudaTensor *density, 
    THCudaTensor *grid, float grid_lowerx, float grid_lowery, float grid_lowerz, float grid_stepsx,
    float grid_stepsy, float grid_stepsz, float radius)
{
    /*locs = THCudaTensor_newContiguous(state, locs);
    data = THCudaTensor_newContiguous(state, data);
    density = THCudaTensor_newContiguous(state, density);
    grid = THCudaTensor_newContiguous(state, grid);*/

    float* points = THCudaTensor_data(state, locs);
    float* pdata = THCudaTensor_data(state, data);
    float* pdensity = THCudaTensor_data(state, density);
    int batch_size = locs->size[0];
    int nparticles = locs->size[1];
    int data_dims = data->size[2];
    int grid_dimsx = grid->size[1];
    int grid_dimsy = grid->size[2];
    int grid_dimsz = grid->size[3];
    float* pgrid = THCudaTensor_data(state, grid);
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_particles2grid(points, pdata, pdensity, nparticles, batch_size, data_dims,
        grid_lowerx, grid_lowery, grid_lowerz, grid_dimsx, grid_dimsy, grid_dimsz,
        grid_stepsx, grid_stepsy, grid_stepsz, pgrid, radius, stream);
}


/* Grid2Particles Layer */

int spnc_grid2particles_forward_cuda(THCudaTensor *grid, float grid_lowerx, float grid_lowery, 
    float grid_lowerz, float grid_stepsx, float grid_stepsy, float grid_stepsz, 
    THCudaTensor *locs, THCudaTensor *data)
{
    /*grid = THCudaTensor_newContiguous(state, grid);
    locs = THCudaTensor_newContiguous(state, locs);
    data = THCudaTensor_newContiguous(state, data);*/

    float* points = THCudaTensor_data(state, locs);
    float* pdata = THCudaTensor_data(state, data);
    int batch_size = locs->size[0];
    int nparticles = locs->size[1];
    int data_dims = data->size[2];
    int grid_dimsx = grid->size[1];
    int grid_dimsy = grid->size[2];
    int grid_dimsz = grid->size[3];
    float* pgrid = THCudaTensor_data(state, grid);
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_grid2particles(pgrid, batch_size,
        grid_lowerx, grid_lowery, grid_lowerz, grid_dimsx, grid_dimsy, grid_dimsz,
        grid_stepsx, grid_stepsy, grid_stepsz, data_dims, points, nparticles, pdata, stream);
}

int spnc_grid2particles_backward_cuda(THCudaTensor *locs, THCudaTensor *ddata, THCudaTensor *dgrid, 
    float grid_lowerx, float grid_lowery, float grid_lowerz, float grid_stepsx, float grid_stepsy, 
    float grid_stepsz)
{
    /*dgrid = THCudaTensor_newContiguous(state, dgrid);
    locs = THCudaTensor_newContiguous(state, locs);
    ddata = THCudaTensor_newContiguous(state, ddata);*/

    float* points = THCudaTensor_data(state, locs);
    float* pddata = THCudaTensor_data(state, ddata);
    int batch_size = locs->size[0];
    int nparticles = locs->size[1];
    int data_dims = ddata->size[2];
    int grid_dimsx = dgrid->size[1];
    int grid_dimsy = dgrid->size[2];
    int grid_dimsz = dgrid->size[3];
    float* pdgrid = THCudaTensor_data(state, dgrid);
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_grid2particles_backward(points, pddata, batch_size, nparticles, data_dims, pdgrid,
        grid_lowerx, grid_lowery, grid_lowerz, grid_dimsx, grid_dimsy, grid_dimsz,
        grid_stepsx, grid_stepsy, grid_stepsz, stream);
}



/* SDFs2Grid Layer */

int spnc_sdfs2grid_forward_cuda(THCudaTensor* sdfs, THCudaIntTensor* sdf_shapes, THCudaIntTensor* indices,
    THCudaTensor* sdf_poses, THCudaTensor* sdf_widths, THCudaTensor* grid, float grid_lowerx, float grid_lowery, 
    float grid_lowerz, float grid_stepsx, float grid_stepsy, float grid_stepsz)
{
    /*grid = THCudaTensor_newContiguous(state, grid);
    sdfs = THCudaTensor_newContiguous(state, sdfs);
    indices = THCudaIntTensor_newContiguous(state, indices);
    sdf_poses = THCudaTensor_newContiguous(state, sdf_poses);
    sdf_widths = THCudaTensor_newContiguous(state, sdf_widths);
    sdf_shapes = THCudaIntTensor_newContiguous(state, sdf_shapes);*/

    float* psdfs = THCudaTensor_data(state, sdfs);
    int* sdf_dims = THCudaIntTensor_data(state, sdf_shapes);
    int stride_per_sdf = sdfs->size[1];
    int nsdfs = sdfs->size[0];
    int* sdf_indices = THCudaIntTensor_data(state, indices);
    int batch_size = indices->size[0];
    int nsdf_indices = indices->size[1];
    float* psdf_poses = THCudaTensor_data(state, sdf_poses);
    float* psdf_widths = THCudaTensor_data(state, sdf_widths);
    float* pgrid = THCudaTensor_data(state, grid);
    int grid_dimsx = grid->size[1];
    int grid_dimsy = grid->size[2];
    int grid_dimsz = grid->size[3];
    cudaStream_t stream = THCState_getCurrentStream(state);

    return cuda_forward_sdfs2grid(psdfs, sdf_dims, stride_per_sdf, nsdfs, sdf_indices, batch_size, nsdf_indices, 
                psdf_poses, psdf_widths, pgrid, grid_lowerx, grid_lowery, grid_lowerz, grid_dimsx, grid_dimsy, 
                grid_dimsz, grid_stepsx, grid_stepsy, grid_stepsz, stream);
}



