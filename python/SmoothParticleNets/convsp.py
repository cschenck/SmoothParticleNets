
import numbers
import numpy as np

import torch
import torch.autograd

import _ext


class ConvSP(torch.nn.Module):
    """ The Smooth Particle Convolution layer. Performs convolutions on particle sets. Each
    particle has a location in N-D space and an associated set of features. An N-D kernel 
    is centered at each particle location, with the shape of the kernel and the size of 
    each kernel cell being user-specified. Then the feature field induced by the set of
    particles is evaluated at each kernel cell. That is, for every feature f associated 
    with the particles and kernel cell center r, a weighted average for f is taken at r
    based on the distance to nearby
    """
    def __init__(self, in_channels, out_channels, ndim, kernel_size, dilation, radius):
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
        """
        super(ConvSP, self).__init__()
        self.nchannels = _check_conditions(in_channels, "in_channels", 
            "%s > 0", "isinstance(%s, numbers.Integral)")
        self.nkernels = _check_conditions(out_channels, "out_channels",
            "%s > 0", "isinstance(%s, numbers.Integral)")
        self.ndim = _check_conditions(ndim, "ndim", 
            "%s > 0", "%s < " + str(_ext.spn_max_cartesian_dim()), 
            "isinstance(%s, numbers.Integral)")

        self.kernel_size = _make_list(kernel_size, ndim, "kernel_size", 
            "%s >= 0", "%s %% 2 == 1 # Must be odd", 
            "isinstance(%s, numbers.Integral)")
        self.dilation = _make_list(dilation, ndim, "dilation", 
            "%s >= 0", "isinstance(%s, numbers.Real)")

        self.radius = _check_conditions(radius, "radius", 
            "%s >= 0", "isinstance(%s, numbers.Real)")

        self.ncells = np.prod(self.kernel_size)
        self.weight = torch.nn.Parameter(torch.Tensor(self.nkernels, self.nchannels, self.ncells))
        self.bias = torch.nn.Parameter(torch.Tensor(self.nkernels))

        self.kernel_size = _list2tensor(self.kernel_size)
        self.dilation = _list2tensor(self.dilation)

    def forward(self, locs, data, density):
        """ Compute a forward pass of the Smooth Particle Convolution Layer.

        Inputs:
            -locs: A BxNx(D+1) tensor where B is the batch size, N is the number
                   of particles, and D is the dimensionality of the particles'
                   coordinate space. The last element in the D+1 dimension should
                   be the inverse mass of the particle.
            -data: A BxCxN tensor where C is the number of input features.
            -density: A BxN tensor with the density at each particle. If you need
                      to compute the density, you can instantiate a version of
                      this layer with, kernel_size=in_channels=out_channels=1, 
                      set the weights to 1 and the bias to 0. Then call forward
                      with data and density filled with 1s. The result will be
                      the density.

        Returns: A BxOxN tensor where O is the number of output features.
        """

        # Error checking.
        batch_size = locs.size()[0]
        N = locs.size()[1]
        _check_tensor_dims(locs, "locs", (batch_size, N, self.ndim + 1))
        _check_tensor_dims(data, "data", (batch_size, self.nchannels, N))
        _check_tensor_dims(density, "density", (batch_size, N))

        # Do the compution.
        convsp = _ConvSPFunction(self.radius, self.kernel_size, self.dilation,
            self.ncells)
        # data.shape = BxCxN
        data = convsp(locs, data, density, self.weight, self.bias)
        # data.shape = BxOxN
        return data



"""

INTERNAL FUNCTIONS

"""

class _ConvSPFunction(torch.autograd.Function):

    def __init__(self, radius, kernel_size, dilation, ncells):
        super(_ConvSPFunction, self).__init__()
        self.radius = radius
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.ncells = ncells

    def forward(self, locs, data, density, weight, bias):
        self.save_for_backward(locs, data, density, weight, bias)
        batch_size = locs.size()[0]
        N = locs.size()[1]
        nkernels = weight.size()[0]
        ret = data.new(batch_size, nkernels, N)
        ret.fill_(0)
        if locs.is_cuda:
            if not _ext.spnc_convsp_forward(locs, data, density, weight, bias, self.radius, 
                        self.kernel_size.cuda(), self.dilation.cuda(), ret):
                raise Exception("Cuda error")
        else:
            _ext.spn_convsp_forward(locs, data, density, weight, bias, self.radius, 
                self.kernel_size, self.dilation, ret)

        # Add the bias.
        ret += bias.view(1, nkernels, 1)

        return ret 


    def backward(self, grad_output):
        locs, data, density, weight, bias = self.saved_tensors
        ret_data = grad_output.new(data.size())
        ret_data.fill_(0)
        ret_weight = grad_output.new(weight.size())
        ret_weight.fill_(0)
        if grad_output.is_cuda:
            if not _ext.spnc_convsp_backward(locs, data, density, weight, bias, self.radius, 
                        self.kernel_size.cuda(), self.dilation.cuda(), grad_output, ret_data,
                        ret_weight):
                raise Exception("Cuda error")
        else:
            _ext.spn_convsp_backward(locs, data, density, weight, bias, self.radius, 
                self.kernel_size, self.dilation, grad_output, ret_data, ret_weight)

        # PyTorch requires gradients for each input, but we only care about the
        # gradients for data, so just set the rest to 0.
        return (grad_output.new(locs.size()).fill_(0), 
                ret_data, 
                grad_output.new(density.size()).fill_(0),
                ret_weight,
                grad_output.sum(2).sum(0))




def _throws_exception(exception_type, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
        return False
    except exception_type:
        return True

def _check_conditions(v, name, *conditions):
    for condition in conditions:
        if not eval(condition % "v"):
            raise ValueError(("%s must meet the following condition: " + condition) % (name, name))
    return v

def _make_list(l, length, name, *conditions):
    if _throws_exception(TypeError, list, l):
        l = [l]*length
    else:
        l = list(l)
    if len(l) != length:
        raise ValueError("%s must be a list of length %d." % (name, length))
    for i, ll in enumerate(l):
        l[i] = _check_conditions(ll, name, *conditions)
    return l

def _check_tensor_dims(t, name, dims):
    s = t.size()
    if len(s) != len(dims):
        raise ValueError("%s must be a %d-dimensional tensor." % (name, len(dims)))
    for i in range(len(dims)):
        if s[i] != dims[i]:
            raise ValueError("The %dth dimension of %s must have size %d, not %d." % (i, name, dims[i], s[i]))

def _list2tensor(l):
    return torch.from_numpy(np.array(l, dtype=np.float32))