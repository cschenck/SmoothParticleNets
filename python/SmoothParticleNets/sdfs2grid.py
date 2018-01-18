

import torch
import torch.autograd

import _ext


class SDFs2GridFunction(torch.autograd.Function):

    def __init__(self, grid_shape, grid_lower, grid_steps):
        self.grid_lower = grid_lower
        self.grid_steps = grid_steps
        self.grid_shape = grid_shape

    def forward(self, sdfs, sdf_shapes, indices, sdf_poses, sdf_widths):
        has_batch = True
        if len(indices.size()) == 1: # No batch size included.
            indices = indices.unsqueeze(0)
            sdf_widths = sdf_widths.unsqueeze(0)
            sdf_poses = sdf_poses.unsqueeze(0)
            has_batch = False
        
        # Error checking.
        if len(indices.size()) != 2:
            raise ValueError("indices must be a 1 or 2-D tensor.")
        if len(sdf_widths.size()) != 3:
            raise ValueError("sdf_widths must be a 2 or 3-D tensor.")
        if len(sdf_poses.size()) != 3:
            raise ValueError("sdf_poses must be a 2 or 3-D tensor.")
        if sdf_widths.size()[0] != indices.size()[0]:
            raise ValueError("sdf_widths and indices must have the same batch size.")
        if sdf_poses.size()[0] != indices.size()[0]:
            raise ValueError("sdf_poses and indicies must have the same batch size.")
        if sdf_widths.size()[1] != indices.size()[1]:
            raise ValueError("sdf_widths and indicies must have the same number of sdfs.")
        if sdf_poses.size()[1] != indices.size()[1]:
            raise ValueError("sdf_poses and indices must have the same number of sdfs.")
        if sdf_widths.size()[2] != 3:
            raise ValueError("sdf_widths last dimension must have size 3 (the size of x, y, and z).")
        if sdf_poses.size()[2] != 7:
            raise ValueError("sdf_poses last dimension must have size 7 (xyz translation and quaternion).")
        if len(sdfs.size()) != 2:
            raise ValueError("sdfs must be a 2D tensor (num sdfs X unrolled SDF data).")
        if len(sdf_shapes.size()) != 2:
            raise ValueError("sdf_shapes must be a 2D tensor.")
        if sdfs.size()[0] != sdf_shapes.size()[0]:
            raise ValueError("sdfs and sdf_shapes must have the same number of sdfs.")

        grid = sdfs.new((indices.size()[0],) + self.grid_shape)
        if locs.is_cuda:
            _ext.spnc_sdfs2grid_forward_cuda(sdfs, sdf_shapes, indices, sdf_poses, sdf_widths, grid, 
                self.grid_lower[0], self.grid_lower[1], self.grid_lower[2], self.grid_steps[0],
                self.grid_steps[1], self.grid_steps[2])
        else:
            raise NotImplementedError("SDFs2Grid forward is only implemented on the GPU (for now).")
        
        if not has_batch:
            grid = grid.squeeze(0)
        return grid


    def backward(self, grad_output):
        raise NotImplementedError("SDFs2Grid backward is not implemented.")



class SDFs2Grid(torch.nn.Module):
    def __init__(self, grid_shape, grid_lower, grid_steps):
        super(SDFs2Grid, self).__init__()
        self.grid_shape = grid_shape
        self.grid_lower = grid_lower
        self.grid_steps = grid_steps

    def forward(self, sdfs, sdf_shapes, indices, sdf_poses, sdf_widths):
        return SDFs2GridFunction(self.grid_shape, self.grid_lower, self.grid_steps)(
            sdfs, sdf_shapes, indices, sdf_poses, sdf_widths)
