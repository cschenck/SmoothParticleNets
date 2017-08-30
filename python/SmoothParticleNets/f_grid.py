

import torch
import torch.autograd

import _ext


class FGridFunction(torch.autograd.Function):

    def forward(self, locs, value_grid):
        self.save_for_backward(locs, value_grid)
        s = locs.size()
        has_batch = True
        if len(s) == 2: # No batch size included.
            locs = locs.unsqueeze(0)
            value_grid = value_grid.unsqueeze(0)
            s = locs.size()
            has_batch = False
        elif len(s) != 3:
            raise ValueError("Locs must be a 2 or 3-D tensor.")

        d = value_grid.size()
        if d[0] != s[0]:
            raise ValueError("locs and value_grid must have the same batch size.")

        if len(d) < s[-1] + 1:
            raise ValueError("The dimensionality of the value grid must be at least the size of the last dimension in locs pluse 1.")

        ret = locs.new(*(s[:2] + d[(s[-1] + 1):]))
        if locs.is_cuda:
            _ext.spnc_fgrid_forward_cuda(locs, value_grid, ret)
        else:
            # Do this the slow way since we only don't use cuda when debugging.
            for b in range(s[0]):
                for i in range(s[1]):
                    if all(k >= 0 and k < d[j+1] for j, k in enumerate(locs[b, i, :])):
                        p = [b] + locs[b, i, :].tolist()
                        p = tuple(int(pp) for pp in p)
                        if len(d) > s[-1] + 1:
                            ret[b, i, ...] = value_grid[p, ...]
                        else:
                            ret[b, i] = value_grid[p]
                    else:
                        ret[b, i, ...] = 0.0
        
        if not has_batch:
            ret = ret.squeeze(0)
        return ret


    def backward(self, grad_output):
        locs, value_grid = self.saved_tensors
        s = locs.size()
        has_batch = True
        if len(s) == 2: # No batch size included.
            locs = locs.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)
            value_grid = value_grid.unsqueeze(0)
            s = locs.size()
            has_batch = False

        d = value_grid.size()
        ret = value_grid.new(*d).fill_(0)
        locs_ret = locs.new(*s).fill_(0)
        if grad_output.is_cuda:
            _ext.spnc_fgrid_backward_cuda(locs, grad_output, ret)
        else:
            # Do this the slow way for now.
            for b in range(s[0]):
                for i in range(s[1]):
                    if all(k >= 0 and k < d[j+1] for j, k in enumerate(locs[b, i, :])):
                        p = [b] + locs[b, i, :].tolist()
                        p = tuple(int(pp) for pp in p)
                        if len(d) > s[-1] + 1:
                            ret[p, ...] += grad_output[b, i, ...]
                        else:
                            ret[p] += grad_output[b, i]
        if not has_batch:
            ret = ret.squeeze(0)
            locs_ret = locs_ret.squeeze(0)
        return locs_ret, ret



class FGrid(torch.nn.Module):
    def __init__(self):
        super(FGrid, self).__init__()

    def forward(self, locs, value_grid):
        return FGridFunction()(locs, value_grid)
