

import torch
import torch.autograd

import _ext


class Grid2ParticlesFunction(torch.autograd.Function):

    def __init__(self, grid_lower, grid_steps):
        self.grid_lower = grid_lower
        self.grid_steps = grid_steps

    def forward(self, grid, locs):
        self.save_for_backward(grid, locs)
        s = locs.size()
        has_batch = True
        if len(s) == 2: # No batch size included.
            locs = locs.unsqueeze(0)
            grid = grid.unsqueeze(0)
            s = locs.size()
            has_batch = False
        elif len(s) != 3:
            raise ValueError("Locs must be a 2 or 3-D tensor.")

        if s[-1] != 4:
            raise ValueError("The last dimension of locs must have size 4: xyzw, where w is a placeholder for the inverse mass.")
        if grid.size()[0] != s[0]:
            raise ValueError("locs and data must have the same batch size.")

        ds = (s[0], s[1])
        if len(grid.size()) > 4:
            ds = ds + (grid.size()[4],)

        ret = grid.new(*ds)
        if locs.is_cuda:
            _ext.spnc_grid2particles_forward_cuda(grid, self.grid_lower[0], self.grid_lower[1], self.grid_lower[2],
                self.grid_steps[0], self.grid_steps[1], self.grid_steps[2], locs, ret)
        else:
            raise NotImplementedError("Grid2Particles forward is only implemented on the GPU (for now).")
        
        if not has_batch:
            ret = ret.squeeze(0)
        return ret


    def backward(self, grad_output):
        grid, locs = self.saved_tensors
        s = locs.size()
        has_batch = True
        if len(s) == 2: # No batch size included.
            locs = locs.unsqueeze(0)
            grid = grid.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)
            s = locs.size()
            has_batch = False

        ret = grid.new(grid.size())
        locs_ret = locs.new(locs.size()).fill_(0)
        if locs.is_cuda:
            _ext.spnc_grid2particles_backward_cuda(locs, grad_output, ret, self.grid_lower[0], self.grid_lower[1],
                self.grid_lower[2], self.grid_steps[0], self.grid_steps[1], self.grid_steps[2])
        else:
            raise NotImplementedError("Grid2Particles backward is only implemented on the GPU (for now).")
        
        if not has_batch:
            ret = ret.squeeze(0)
            locs_ret = locs_ret.squeeze(0)
        return ret, locs_ret



class Grid2Particles(torch.nn.Module):
    def __init__(self, grid_lower, grid_steps):
        super(Grid2Particles, self).__init__()
        self.grid_lower = grid_lower
        self.grid_steps = grid_steps

    def forward(self, grid, locs):
        return Grid2ParticlesFunction(self.grid_lower, self.grid_steps)(grid, locs)
