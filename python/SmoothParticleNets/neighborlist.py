
import numbers
import numpy as np

import torch
import torch.autograd

import _ext
import error_checking as ec


class NeighborList(torch.nn.Module):
    """ TODO
    """
    def __init__(self, radius):
        """ TODO
        """
        super(NeighborList, self).__init__()
        self.radius = radius

    def forward(self, locs):
        """ TODO
        """

        fn = _NeighborListFunction(self.radius)
        return fn(locs)



"""

INTERNAL FUNCTIONS

"""

class _NeighborListFunction(torch.autograd.Function):

    def __init__(self, radius):
        super(_NeighborListFunction, self).__init__()
        self.radius = radius

    def forward(self, locs):
        batch_size = locs.size()[0]
        N = locs.size()[1]
        ret = locs.new(batch_size, N, int(np.ceil(N/(locs.element_size()*8.0))))
        ret.fill_(0)
        if locs.is_cuda:
            if not _ext.spnc_neighborlist(locs, self.radius, ret):
                raise Exception("Cuda error")
        else:
            _ext.spn_neighborlist(locs, self.radius, ret)

        return ret 


    def backward(self, grad_output):
        raise NotImplementedError("Backward is not implemented for NeighborList.")




