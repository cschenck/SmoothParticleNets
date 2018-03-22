
import numbers
import numpy as np

import torch
import torch.autograd

import _ext
import error_checking as ec

class ParticleCollision(torch.nn.Module):
    """ TODO
    """
    def __init__(self, ndim, radius, max_grid_dim=96, max_collisions=128):
        """ Initialize a Particle Collision layer.

        Arguments:
            -ndim: The dimensionality of the particle's coordinate space.
            -radius: The radius to use when computing the neighbors for each query point.
            -max_grid_dim: The maximum size of all the dimensions for the internal hash
                           grid. Set this value lower if you are running out of memory.
            -max_collisions: The maximum number of neighbors a particle may have.
        """
        super(ParticleCollision, self).__init__()
        self.ndim = ec.check_conditions(ndim, "ndim", 
            "%s > 0", "%s < " + str(_ext.spn_max_cartesian_dim()), 
            "isinstance(%s, numbers.Integral)")

        self.radius = ec.check_conditions(radius, "radius", 
            "%s >= 0", "isinstance(%s, numbers.Real)")

        self.max_grid_dim = ec.check_conditions(max_grid_dim, "max_grid_dim", 
            "%s > 0", "isinstance(%s, numbers.Integral)")

        self.max_collisions = ec.check_conditions(max_collisions, "max_collisions", 
            "%s > 0", "isinstance(%s, numbers.Integral)")

        self.radixsort_buffer_size = -1

        self.register_buffer("cellIDs", torch.zeros(1, 1))
        self.register_buffer("cellStarts", torch.zeros(1, max_grid_dim**ndim))
        self.register_buffer("cellEnds", torch.zeros(1, max_grid_dim**ndim))
        self.register_buffer("cuda_buffer", torch.zeros(1,))

    def forward(self, locs, data=None):
        """ Compute the neighbors of each location. Reorders the locs and data tensors
        in place and returns the list of indices in their new order and the list of
        neighbors for each location.

        Inputs:
            -locs: A BxNxD tensor where B is the batch size, N is the number
                   of particles, and D is the dimensionality of the particles'
                   coordinate space.
            -data: [optional] A BxNxC tensor where C is the number of channels.

        Returns: 
            -Idxs: BxN tensor with the original index of each location in their
                   new order.
            -Neighbors: BxNxM where M is max_neighbors. This lists the indices of
                        all locations within radius of each location, up to M. If
                        there are fewer than M neighbors, -1 is used to indicate
                        the end of the neighbor list.
        """

        # Error checking.
        batch_size = locs.size()[0]
        N = locs.size()[1]
        ec.check_tensor_dims(locs, "locs", (batch_size, N, self.ndim))
        if data is not None:
            ec.check_tensor_dims(data, "data", (batch_size, N, -1))
            data = data.contiguous()
        else:
            data = Variable(locs.data.new(0, 0, 0), requires_grad=False)

        locs = locs.contiguous()

        # Resize the internal buffers to be the right size.
        buffers = [self.cellIDs, self.cellStarts, self.cellEnds]
        for buf in buffers:
            if buf.size()[0] != batch_size:
                ns = (batch_size,) + buf.size()[1:]
                buf.resize_(ns)
        if self.cellIDs.size()[1] != N:
            # Allocate 2 extra batches on cellIDs for sorting.
            self.cellIDs.resize_(batch_size + 2, N)

        if locs.is_cuda:
            if self.radixsort_buffer_size < 0:
                self.radixsort_buffer_size = _ext.spnc_get_radixsort_buffer_size()
            bufsize = max(self.radixsort_buffer_size, 
                np.prod(locs.size()) + np.prod(data.size()))
            if self.cuda_buffer.size()[0] != bufsize:
                self.cuda_buffer.resize_(bufsize)

        # Compute grid bounds.
        lower_bounds, _ = locs.min(1)
        upper_bounds, _ = locs.max(1)
        grid_dims = torch.ceil(torch.clamp((upper_bounds - lower_bounds)/self.radius, 
            0, self.max_grid_dim))
        center = (lower_bounds + upper_bounds)/2
        lower_bounds = center - grid_dims*self.radius/2

        # Do the compution.
        coll = _ParticleCollisionFunction(self.radius, self.max_grid_dim, self.max_collisions,
            self.cellIDs, self.cellStarts, self.cellEnds, self.cuda_buffer)
        idxs, neighbors = coll(locs, data, lower_bounds, grid_dims)
        return idxs, neighbors



"""

INTERNAL FUNCTIONS

"""

class _ParticleCollisionFunction(torch.autograd.Function):

    def __init__(self, radius, max_grid_dim, max_collisions, cellIDs, cellStarts, cellEnds,
                    cuda_buffer):
        super(_ParticleCollisionFunction, self).__init__()
        self.radius = radius
        self.max_grid_dim = max_grid_dim
        self.max_collisions = max_collisions
        self.cellIDs = cellIDs
        self.cellStarts = cellStarts
        self.cellEnds = cellEnds
        self.cuda_buffer = cuda_buffer

    def forward(self, locs, data, lower_bounds, grid_dims):
        batch_size = locs.size()[0]
        N = locs.size()[1]
        idxs = locs.new(batch_size, N)
        neighbors = locs.new(batch_size, N, self.max_collisions)
        neighbors.fill_(-1)
        if locs.is_cuda:
            if not _ext.spnc_compute_collisions(locs, data, lower_bounds, grid_dims, 
                    self.cellIDs, idxs, self.cellStarts, self.cellEnds, neighbors, 
                    self.cuda_buffer, self.radius, self.radius):
                raise Exception("Cuda error")
        else:
            _ext.spn_compute_collisions(locs, data, lower_bounds, grid_dims, 
                self.cellIDs, idxs, self.cellStarts, self.cellEnds, neighbors, 
                self.radius, self.radius)

        return idxs, neighbors 




