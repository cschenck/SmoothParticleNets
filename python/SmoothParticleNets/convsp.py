
import numbers
import numpy as np

import torch
import torch.autograd

import _ext
import error_checking as ec
from kernels import KERNELS, KERNEL_NAMES

class ConvSP(torch.nn.Module):
    """ The Smooth Particle Convolution layer. Performs convolutions on particle sets. Each
    particle has a location in N-D space and an associated set of features. An N-D kernel 
    is centered at each particle location, with the shape of the kernel and the size of 
    each kernel cell being user-specified. Then the feature field induced by the set of
    particles is evaluated at each kernel cell. That is, for every feature f associated 
    with the particles and kernel cell center r, a weighted average for f is taken at r
    based on the distance to nearby
    """
    def __init__(self, in_channels, out_channels, ndim, kernel_size, dilation, radius,
                    dis_norm=False, kernel_fn='poly', diffdata=False, with_params=True):
        """ Initialize a Smooth Particle Convolution layer.

        Arguments:
            -in_channels: The number of features for each input particle.
            -out_channels: The number of features to output for each particle.
            -ndim: The dimensionality of the particle's coordinate space.
            -kernel_size: (int or tuple) The shape of the kernel that is place around each
                          particle. The kernel is centered on the particle, so the size
                          must be odd.
            -dilation: (float or tuple) The spacing between each cell of the kernel.
            -radius: The radius to use when computing the neighbors for each query point.
            -kernel_fn: The kernel function to use in the SPH equation. Refer to kernels.py
                        for a list and explanation of all available functions.
            -dis_norm: If true, will divide by the particle-to-particle distance in the
                       SPH equation.
            -diffdata: If true, will take the difference between the data of each particle
                       and the query particle's data, e.g., instead of the result being
                       \sum_j [d_j * W(d_ij)], it would be \sum_j [(d_j - d_i) * W(d_ij)]
                       where d_k is the data associated with particle k.
            -with_params: If true, the parameters weight and bias are registered with
                          PyTorch as parameters. Otherwise they are registered as buffers,
                          meaning they won't be optimized when doing backprop.
        """
        super(ConvSP, self).__init__()
        self.nchannels = ec.check_conditions(in_channels, "in_channels", 
            "%s > 0", "isinstance(%s, numbers.Integral)")
        self.nkernels = ec.check_conditions(out_channels, "out_channels",
            "%s > 0", "isinstance(%s, numbers.Integral)")
        self.ndim = ec.check_conditions(ndim, "ndim", 
            "%s > 0", "%s < " + str(_ext.spn_max_cartesian_dim()), 
            "isinstance(%s, numbers.Integral)")

        self._kernel_size = ec.make_list(kernel_size, ndim, "kernel_size", 
            "%s >= 0", "%s %% 2 == 1 # Must be odd", 
            "isinstance(%s, numbers.Integral)")
        self._dilation = ec.make_list(dilation, ndim, "dilation", 
            "%s >= 0", "isinstance(%s, numbers.Real)")

        self.radius = ec.check_conditions(radius, "radius", 
            "%s >= 0", "isinstance(%s, numbers.Real)")

        self.kernel_fn = ec.check_conditions(kernel_fn, "kernel_fn",
            "%s in " + str(KERNEL_NAMES))
        self.kernel_fn = KERNEL_NAMES.index(self.kernel_fn)
        self.dis_norm = (1 if dis_norm else 0)
        self.diffdata = (1 if diffdata else 0)

        self.ncells = np.prod(self._kernel_size)

        if with_params:
            self.register_parameter("weight", torch.nn.Parameter(torch.Tensor(self.nkernels, 
                self.nchannels, self.ncells)))
            self.register_parameter("bias", torch.nn.Parameter(torch.Tensor(self.nkernels)))
        else:
            self.register_buffer("weight", torch.autograd.Variable(torch.Tensor(self.nkernels, 
                self.nchannels, self.ncells)))
            self.register_buffer("bias", torch.autograd.Variable(torch.Tensor(self.nkernels)))

        self.register_buffer("kernel_size", ec.list2tensor(self._kernel_size))
        self.register_buffer("dilation", ec.list2tensor(self._dilation))

        self.nshared_device_mem = -1
        self.device_id = -1

    def forward(self, locs, data, neighbors):
        """ Compute a forward pass of the Smooth Particle Convolution Layer.

        Inputs:
            -locs: A BxNxD tensor where B is the batch size, N is the number
                   of particles, and D is the dimensionality of the particles'
                   coordinate space.
            -data: A BxNxC tensor where C is the number of input features.
            -neighbors: A BxNxM tensor where M is the maximum number of neighbors.
                        For each particle, this is the list of particle indices
                        that are considered neighbors. If the number of neighbors
                        is less than M, the less is ended with a -1.

        Returns: A BxNxO tensor where O is the number of output features.
        """

        # Error checking.
        batch_size = locs.size()[0]
        N = locs.size()[1]
        ec.check_tensor_dims(locs, "locs", (batch_size, N, self.ndim))
        ec.check_tensor_dims(data, "data", (batch_size, N, self.nchannels))

        locs = locs.contiguous()
        data = data.contiguous()

        if locs.is_cuda:
            if self.device_id != torch.cuda.current_device():
                self.device_id = torch.cuda.current_device()
                self.nshared_device_mem = _ext.spnc_get_shared_mem_size(self.device_id)

        # Do the compution.
        convsp = _ConvSPFunction(self.radius, self.kernel_size, self.dilation,
            self.dis_norm, self.kernel_fn, self.diffdata, self.ncells, self.nshared_device_mem)
        # data.shape = BxCxN
        data = convsp(locs, data, neighbors, self.weight, self.bias)
        # data.shape = BxOxN
        return data



"""

INTERNAL FUNCTIONS

"""

class _ConvSPFunction(torch.autograd.Function):

    def __init__(self, radius, kernel_size, dilation, dis_norm, kernel_fn, diffdata,
            ncells, nshared_device_mem=-1):
        super(_ConvSPFunction, self).__init__()
        self.radius = radius
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dis_norm = dis_norm
        self.kernel_fn = kernel_fn
        self.diffdata = diffdata
        self.ncells = ncells
        self.nshared_device_mem = nshared_device_mem

    def forward(self, locs, data, neighbors, weight, bias):
        self.save_for_backward(locs, data, neighbors, weight, bias)
        batch_size = locs.size()[0]
        N = locs.size()[1]
        nkernels = weight.size()[0]
        ret = data.new(batch_size, N, nkernels)
        ret.fill_(0)
        if locs.is_cuda:
            if not _ext.spnc_convsp_forward(locs, data, neighbors, weight, bias, self.radius, 
                        self.kernel_size, self.dilation, self.dis_norm, self.kernel_fn, self.diffdata,
                        ret, self.nshared_device_mem):
                raise Exception("Cuda error")
        else:
            _ext.spn_convsp_forward(locs, data, neighbors, weight, bias, self.radius, 
                self.kernel_size, self.dilation, self.dis_norm, self.kernel_fn, self.diffdata, ret)

        # Add the bias.
        ret += bias.view(1, 1, nkernels)

        return ret 


    def backward(self, grad_output):
        locs, data, neighbors, weight, bias = self.saved_tensors
        ret_data = grad_output.new(data.size())
        ret_data.fill_(0)
        ret_weight = grad_output.new(weight.size())
        ret_weight.fill_(0)
        if grad_output.is_cuda:
            if not _ext.spnc_convsp_backward(locs, data, neighbors, weight, bias, self.radius, 
                        self.kernel_size, self.dilation, self.dis_norm, self.kernel_fn, 
                        self.diffdata, grad_output, ret_data, ret_weight, self.nshared_device_mem):
                raise Exception("Cuda error")
        else:
            _ext.spn_convsp_backward(locs, data, neighbors, weight, bias, self.radius, 
                self.kernel_size, self.dilation, self.dis_norm, self.kernel_fn, self.diffdata,
                grad_output, ret_data, ret_weight)

        # PyTorch requires gradients for each input, but we only care about the
        # gradients for data, so just set the rest to 0.
        return (grad_output.new(locs.size()).fill_(0), 
                ret_data, 
                grad_output.new(neighbors.size()).fill_(0),
                ret_weight,
                grad_output.sum(1).sum(0))




