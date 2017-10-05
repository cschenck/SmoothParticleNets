

import torch
import torch.autograd

import _ext


class Grid2ParticlesFunction(torch.autograd.Function):

    def __init__(self, grid_lower, grid_steps):
        self.grid_lower = grid_lower
        self.grid_steps = grid_steps

    def forward(self, grid, locs):
        s = locs.size()
        has_batch = True
        if len(s) == 2: # No batch size included.
            locs = locs.unsqueeze(0)
            grid = grid.unsqueeze(0)
            s = locs.size()
            has_batch = False
        elif len(s) != 3:
            raise ValueError("Locs must be a 2 or 3-D tensor.")

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
        raise NotImplementedError("Backwards for the Grid2Particles layer has not been implemented.")



class Grid2Particles(torch.nn.Module):
    def __init__(self, grid_lower, grid_steps):
        super(Grid2Particles, self).__init__()
        self.grid_lower = grid_lower
        self.grid_steps = grid_steps

    def forward(self, grid, locs):
        return Grid2ParticlesFunction(self.grid_lower, self.grid_steps)(grid, locs)
