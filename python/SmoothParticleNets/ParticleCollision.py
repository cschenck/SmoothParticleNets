
import numbers
import numpy as np

import torch
import torch.autograd

import _ext
import _extc
import error_checking as ec


class ReorderData(torch.nn.Module):
    """ TODO
    """

    def __init__(self, reverse=False):
        """ TODO
        if reverse: ret[idxs] = input
        else: ret = input[idxs]
        """
        super(ReorderData, self).__init__()
        self.reverse = (1 if reverse else 0)

    def forward(self, idxs, locs, data=None):
        """ TODO

        Inputs:
            -locs: A BxNxD tensor where B is the batch size, N is the number
                   of particles, and D is the dimensionality of the particles'
                   coordinate space.
            -data: [optional] A BxNxC tensor where C is the number of channels.
        """

        # Error checking.
        batch_size = locs.size()[0]
        N = locs.size()[1]
        ec.check_tensor_dims(locs, "locs", (batch_size, N, -1))
        ec.check_tensor_dims(idxs, "idxs", (batch_size, N))
        if data is not None:
            ec.check_tensor_dims(data, "data", (batch_size, N, -1))
            data = data.contiguous()
            no_data = False
        else:
            data = torch.autograd.Variable(
                locs.data.new(), requires_grad=False)
            no_data = True

        locs = locs.contiguous()
        idxs = idxs.contiguous()

        # Do the compution.
        locs, data = _ReorderDataFunction.apply(idxs, locs, data, self.reverse)
        if no_data:
            return locs
        else:
            return locs, data


class ParticleCollision(torch.nn.Module):
    """ TODO
    """

    def __init__(self, ndim, radius, max_grid_dim=96, max_collisions=128, include_self=True):
        """ Initialize a Particle Collision layer.

        Arguments:
            -ndim: The dimensionality of the particle's coordinate space.
            -radius: The radius to use when computing the neighbors for each query point.
            -max_grid_dim: The maximum size of all the dimensions for the internal hash
                           grid. Set this value lower if you are running out of memory.
            -max_collisions: The maximum number of neighbors a particle may have.
            -include_self: If False, then if the distance between a query location and the
                           particle is 0, that particle will not be included in that query
                           location's neighbor list.
        """
        super(ParticleCollision, self).__init__()
        self.ndim = ec.check_conditions(ndim, "ndim",
                                        "%s > 0", "%s < " +
                                        str(_ext.spn_max_cartesian_dim()),
                                        "isinstance(%s, numbers.Integral)")

        self.radius = ec.check_conditions(radius, "radius",
                                          "%s >= 0", "isinstance(%s, numbers.Real)")

        self.max_grid_dim = ec.check_conditions(max_grid_dim, "max_grid_dim",
                                                "%s > 0", "isinstance(%s, numbers.Integral)")

        self.max_collisions = ec.check_conditions(max_collisions, "max_collisions",
                                                  "%s > 0", "isinstance(%s, numbers.Integral)")

        self.include_self = 1 if include_self else 0

        self.radixsort_buffer_size = -1

        self.register_buffer("cellIDs", torch.zeros(1, 1))
        self.register_buffer("cellStarts", torch.zeros(1, max_grid_dim**ndim))
        self.register_buffer("cellEnds", torch.zeros(1, max_grid_dim**ndim))
        self.register_buffer("cuda_buffer", torch.zeros(1,))

        self.reorder = ReorderData(reverse=False)

    def forward(self, locs, data=None, qlocs=None):
        """ Compute the neighbors of each location. Reorders the locs and data tensors
        in place and returns the list of indices in their new order and the list of
        neighbors for each location.

        Inputs:
            -locs: A BxNxD tensor where B is the batch size, N is the number
                   of particles, and D is the dimensionality of the particles'
                   coordinate space.            
            -data: [optional] A BxNxC tensor where C is the number of channels.
                   Add this to have it reordered alongside locs.
            -qlocs: [optional] A BxMxD tensor of query locations. The neighbors
                    list in the output will be a list of all particles in locs
                    that neighbor each query location. If not provided, locs is
                    used instead.

        Returns: 
            -locs: A BxNxD tensor identical to the input locs, except reordered
                   for optimized memory access.
            -data: [optional] A BxNxC tensor identical to the input data reordered
                   in the same order as locs. If the input data was not provided,
                   then this is not returned.
            -Idxs: BxN tensor with the original index of each particle location in their
                   new order, e.g., idxs[b, i] = j where b is the batch index, j is
                   the original index in locs, and i is the new index.
            -Neighbors: BxMxK where K is max_neighbors. This lists the indices of
                        all particles within radius of each query location, up to K. If
                        there are fewer than K neighbors, -1 is used to indicate
                        the end of the neighbor list. The indices are with respect to
                        the reordered locs tensor. If qlocs is not specified, then
                        locs is used as the query points and it is reordered before
                        being queried, so the neighbors tensor is also reorderd.
        """

        # Error checking.
        batch_size = locs.size()[0]
        N = locs.size()[1]
        ec.check_tensor_dims(locs, "locs", (batch_size, N, self.ndim))
        if data is not None:
            ec.check_tensor_dims(data, "data", (batch_size, N, -1))
            data = data.contiguous()
            has_data = True
        else:
            has_data = False

        if qlocs is not None:
            ec.check_tensor_dims(qlocs, "qlocs", (batch_size, -1, self.ndim))
            qlocs = qlocs.contiguous()

        locs = locs.contiguous()

        # Resize the internal buffers to be the right size.
        buffers = [self.cellIDs, self.cellStarts, self.cellEnds]
        for buf in buffers:
            if buf.size()[0] != batch_size:
                ns = (batch_size,) + buf.size()[1:]
                buf.resize_(ns)
        if self.cellIDs.size()[1] != N or self.cellIDs.size()[0] != batch_size + 2:
            # Allocate 2 extra batches on cellIDs for sorting.
            self.cellIDs.resize_(batch_size + 2, N, 1)

        if locs.is_cuda:
            if self.radixsort_buffer_size < 0:
                self.radixsort_buffer_size = _extc.spnc_get_radixsort_buffer_size()
            bufsize = max(self.radixsort_buffer_size,
                          int(np.prod(locs.size()) + (np.prod(data.size()) if has_data else 0)))
            if self.cuda_buffer.size()[0] != bufsize:
                self.cuda_buffer.resize_(bufsize)

        # Compute grid bounds.
        lower_bounds, _ = locs.min(1)
        upper_bounds, _ = locs.max(1)
        grid_dims = torch.ceil(torch.clamp((upper_bounds - lower_bounds)/self.radius,
                                           0, self.max_grid_dim))
        center = (lower_bounds + upper_bounds)/2
        lower_bounds = center - grid_dims*self.radius/2
        lower_bounds = lower_bounds.contiguous()
        grid_dims = grid_dims.contiguous()

        # Get the new hashgrid order.
        # hashorder = _HashgridOrderFunction(self.radius, self.max_grid_dim, self.cellIDs,
        #                                   self.cuda_buffer)
        # idxs = hashorder(locs, lower_bounds, grid_dims)
        idxs = _HashgridOrderFunction.apply(locs, lower_bounds, grid_dims, self.radius, self.max_grid_dim, 
                self.cellIDs, self.cuda_buffer)

        # Reorder the locs and data.
        if has_data:
            locs, data = self.reorder(idxs, locs, data)
        else:
            locs = self.reorder(idxs, locs)

        # Do the collision compution.
        # coll = _ParticleCollisionFunction(self.radius, self.max_collisions, self.cellIDs,
        #                                  self.cellStarts, self.cellEnds, self.include_self)
        neighbors = _ParticleCollisionFunction.apply(qlocs if qlocs is not None else locs,
                         locs, lower_bounds, grid_dims, self.radius, self.max_collisions, 
                         self.cellIDs, self.cellStarts, self.cellEnds, self.include_self)

        if has_data:
            return locs, data, idxs, neighbors
        else:
            return locs, idxs, neighbors


"""

INTERNAL FUNCTIONS

"""


class _HashgridOrderFunction(torch.autograd.Function):
    def __init__(self, radius, max_grid_dim, cellIDs, cuda_buffer):
        super(_HashgridOrderFunction, self).__init__()
        self.radius = radius
        self.max_grid_dim = max_grid_dim
        self.cellIDs = cellIDs
        self.cuda_buffer = cuda_buffer

    @staticmethod
    def forward(self, locs, lower_bounds, grid_dims, radius, max_grid_dim, cellIDs, cuda_buffer):
        self.save_for_backward(locs, lower_bounds, grid_dims)
        batch_size = locs.size()[0]
        N = locs.size()[1]
        idxs = locs.new(batch_size, N)
        cellIDs.fill_(0)
        if locs.is_cuda:
            if not _extc.spnc_hashgrid_order(locs, lower_bounds, grid_dims,
                                             cellIDs, idxs, cuda_buffer, radius):
                raise Exception("Cuda error")
        else:
            _ext.spn_hashgrid_order(locs, lower_bounds, grid_dims, cellIDs, idxs, radius)

        return idxs

    @staticmethod
    def backward(self, grad_idxs):
        locs, lower_bounds, grid_dims = self.saved_tensors
        return (
            grad_idxs.new(locs.size()).fill_(0),
            grad_idxs.new(lower_bounds.size()).fill_(0),
            grad_idxs.new(grid_dims.size()).fill_(0),)


class _ParticleCollisionFunction(torch.autograd.Function):

    def __init__(self, radius, max_collisions, cellIDs, cellStarts, cellEnds,
                 include_self):
        super(_ParticleCollisionFunction, self).__init__()
        self.radius = radius
        self.max_collisions = max_collisions
        self.cellIDs = cellIDs
        self.cellStarts = cellStarts
        self.cellEnds = cellEnds
        self.include_self = include_self

    @staticmethod
    def forward(self, qlocs, locs, lower_bounds, grid_dims, radius, max_collisions, cellIDs, cellStarts, cellEnds, include_self):
        self.save_for_backward(qlocs, locs, lower_bounds, grid_dims, radius, max_collisions, cellIDs, cellStarts,
                cellEnds, include_self)
        batch_size = locs.size()[0]
        M = qlocs.size()[1]
        neighbors = locs.new(batch_size, M, max_collisions)
        neighbors.fill_(-1)
        cellStarts.fill_(0)
        cellEnds.fill_(0)
        if locs.is_cuda:
            if not _extc.spnc_compute_collisions(qlocs, locs, lower_bounds, grid_dims, cellIDs,
                                                 cellStarts, cellEnds, neighbors, radius, radius,
                                                 include_self):
                raise Exception("Cuda error")
        else:
            _ext.spn_compute_collisions(qlocs, locs, lower_bounds, grid_dims, self.cellIDs,
                                        self.cellStarts, self.cellEnds, neighbors, self.radius, self.radius, self.include_self)

        return neighbors
    
    @staticmethod
    def backward(self, grad_neighbors):
        qlocs, locs, lower_bounds, grid_dims = self.saved_tensors
        return (
            grad_neighbors.new(qlocs.size()).fill_(0),
            grad_neighbors.new(locs.size()).fill_(0),
            grad_neighbors.new(lower_bounds.size()).fill_(0),
            grad_neighbors.new(grid_dims.size()).fill_(0),)


class _ReorderDataFunction(torch.autograd.Function):

    def __init__(self, reverse):
        super(_ReorderDataFunction, self).__init__()
        self.reverse = reverse
    
    @staticmethod
    def forward(self, idxs, locs, data, reverse):
        self.save_for_backward(idxs,reverse)
        nlocs = locs.new(*locs.size())
        ndata = locs.new(*data.size())
        if locs.is_cuda:
            if not _extc.spnc_reorder_data(locs, data, idxs, nlocs, ndata, reverse):
                raise Exception("Cuda error")
        else:
            _ext.spn_reorder_data(locs, data, idxs, nlocs, ndata, reverse)
        return nlocs, ndata
    @staticmethod
    def backward(self, grad_locs, grad_data):
        idxs,reverse = self.saved_tensors
        nlocs = grad_locs.new(*grad_locs.size())
        ndata = grad_data.new(*grad_data.size())
        if grad_locs.is_cuda:
            if not _extc.spnc_reorder_data(grad_locs, grad_data, idxs, nlocs,
                                           ndata, 1 - reverse):
                raise Exception("Cuda error")
        else:
            _ext.spn_reorder_data(grad_locs, grad_data, idxs, nlocs, ndata,
                                  1 - reverse)
        return idxs.new(idxs.size()).fill_(0), nlocs, ndata
