

import torch
import torch.autograd

import _ext


class Particles2GridFunction(torch.autograd.Function):

    def __init__(self, grid_shape, grid_lower, grid_steps, radius):
        self.grid_shape = grid_shape
        self.grid_lower = grid_lower
        self.grid_steps = grid_steps
        self.radius = radius

    def forward(self, locs, data, density):
        s = locs.size()
        has_batch = True
        if len(s) == 2: # No batch size included.
            locs = locs.unsqueeze(0)
            data = data.unsqueeze(0)
            density = density.unsqueeze(0)
            s = locs.size()
            has_batch = False
        elif len(s) != 3:
            raise ValueError("Locs must be a 2 or 3-D tensor.")

        if data.size()[0] != s[0]:
            raise ValueError("locs and data must have the same batch size.")
        if density.size()[0] != s[0]:
            raise ValueError("locs and density must have the same batch size.")
        if data.size()[1] != s[1]:
            raise ValueError("locs and data must have the same number of points.")
        if density.size()[1] != s[1]:
            raise ValueError("locs and density must have the same number of points.")
        if len(density.size()) != 2:
            raise ValueError("density must be a 1 or 2-D tensor.")
        if len(data.size()) != 2 and len(data.size()) != 3:
            raise ValueError("data must be a 1, 2, or 3-D tensor.")

        gs = (s[0],) + tuple(self.grid_shape)
        if len(data.size()) > 2:
            gs = gs + (data.size()[2],)

        ret = data.new(*gs)
        if locs.is_cuda:
            _ext.spnc_particles2grid_forward_cuda(locs, data, density, ret, self.grid_lower[0], 
                self.grid_lower[1], self.grid_lower[2], self.grid_steps[0], self.grid_steps[1], 
                self.grid_steps[2], self.radius)
        else:
            raise NotImplementedError("Particles2Grid forward is only implemented on the GPU (for now).")
        
        if not has_batch:
            ret = ret.squeeze(0)
        return ret


    def backward(self, grad_output):
        raise NotImplementedError("Backwards for the Particles2Grid layer has not been implemented.")



class Particles2Grid(torch.nn.Module):
    def __init__(self, grid_shape, grid_lower, grid_steps, radius):
        super(Particles2Grid, self).__init__()
        self.grid_shape = grid_shape
        self.grid_lower = grid_lower
        self.grid_steps = grid_steps
        self.radius = radius

    def forward(self, locs, data, density):
        return Particles2GridFunction(self.grid_shape, self. grid_lower, self.grid_steps, 
            self.radius)(locs, data, density)
