
import numbers
import numpy as np

import torch
import torch.autograd

import _ext
import _extc
import error_checking as ec


class ConvSDF(torch.nn.Module):
    """ TODO
    """

    def __init__(self, sdfs, sdf_sizes, out_channels, ndim, kernel_size, dilation,
                 max_distance, with_params=True, compute_pose_grads=False):
        """ Initialize a SDF Convolution layer.

        Arguments:
            -sdfs: List of SDFs. Each SDF must be a ndim-dimensional float tensor, with
                   the value of the SDF evaluated at the center of each cell in the grid.
            -sdf_sizes: List of the size of one side of a grid cell in each SDf in sdfs.
            -out_channels: The number of features to output for each query location.
            -ndim: The dimensionality of the coordinate space.
            -kernel_size: (int or tuple) The shape of the kernel that is place around each
                          query location. The kernel is centered on the location, so the 
                          size must be odd.
            -dilation: (float or tuple) The spacing between each cell of the kernel.
            -max_distance: A cap on the maximum SDF value, i.e., the SDF value at any 
                           point p is min(min_i SDF_i(p), max_distance).
            -with_params: If true, the parameters weight and bias are registered with
                          PyTorch as parameters. Otherwise they are registered as buffers,
                          meaning they won't be optimized when doing backprop.
            -compute_pose_grads: If False, will not compute grads wrt to the sdfposes during
                                 backpropagation. This is done to increase speed when
                                 gradients are not neede for the sdf poses.
        """
        super(ConvSDF, self).__init__()
        self.nkernels = ec.check_conditions(out_channels, "out_channels",
                                            "%s > 0", "isinstance(%s, numbers.Integral)")
        self.ndim = ec.check_conditions(ndim, "ndim",
                                        "%s > 0", "%s < " +
                                        str(_ext.spn_max_cartesian_dim()),
                                        "%s in [1, 2, 3] # Only 1-, 2-, and 3-D are suported",
                                        "isinstance(%s, numbers.Integral)")
        self.max_distance = ec.check_conditions(max_distance, "max_distance",
                                                "%s >= 0", "isinstance(%s, numbers.Real)")

        self._kernel_size = ec.make_list(kernel_size, ndim, "kernel_size",
                                         "%s >= 0", "%s %% 2 == 1 # Must be odd",
                                         "isinstance(%s, numbers.Integral)")
        self._dilation = ec.make_list(dilation, ndim, "dilation",
                                      "%s >= 0", "isinstance(%s, numbers.Real)")

        self.register_buffer("sdfs", torch.zeros(1))
        self.register_buffer("sdf_shapes", torch.zeros(1))
        self.register_buffer("sdf_offsets", torch.zeros(1))
        self.SetSDFs(sdfs, sdf_sizes)

        self.ncells = np.prod(self._kernel_size)
        if with_params:
            self.register_parameter("weight", torch.nn.Parameter(torch.Tensor(
                self.nkernels, self.ncells)))
            self.register_parameter(
                "bias", torch.nn.Parameter(torch.Tensor(self.nkernels)))
        else:
            self.register_buffer("weight", torch.autograd.Variable(
                torch.Tensor(self.nkernels, self.ncells)))
            self.register_buffer("bias", torch.autograd.Variable(
                torch.Tensor(self.nkernels)))

        self.compute_pose_grads = (True if compute_pose_grads else False)

        self._kernel_size = ec.list2tensor(self._kernel_size)
        self._dilation = ec.list2tensor(self._dilation)

        self.register_buffer("kernel_size", self._kernel_size)
        self.register_buffer("dilation", self._dilation)

    def SetSDFs(self, sdfs, sdf_sizes):
        cell_sizes = [ec.check_conditions(x, "sdf_sizes[%d]" % i, "%s > 0",
                                          "isinstance(%s, numbers.Real)") for i, x in enumerate(sdf_sizes)]
        _sdfs = [ec.check_conditions(sdf, "sdfs[%d]" % i, "isinstance(%s, torch.Tensor)",
                                     "len(%s.size()) == " + str(self.ndim)) for i, sdf in enumerate(sdfs)]
        _sdf_shapes = ec.list2tensor([list(x.size()) + [cell_sizes[i], ]
                                      for i, x in enumerate(_sdfs)])
        _sdfs = [x.contiguous().view(-1) for x in _sdfs]
        _sdf_offsets = ec.list2tensor([0, ] +
                                      np.cumsum([x.size()[0] for x in _sdfs])[:-1].tolist())
        _sdfs = torch.cat(_sdfs)
        self.sdfs.resize_(_sdfs.size())
        self.sdfs[...] = _sdfs
        self.sdf_shapes.resize_(_sdf_shapes.size())
        self.sdf_shapes[...] = _sdf_shapes
        self.sdf_offsets.resize_(_sdf_offsets.size())
        self.sdf_offsets[...] = _sdf_offsets

    def forward(self, locs, idxs, poses, scales):
        """ Compute a forward pass of the SDF Convolution Layer.

        Inputs:
            -locs: A BxNxD tensor where B is the batch size, N is the number
                   of query locations, and D is the dimensionality of the 
                   coordinate space. 
            -idxs: A BxM tensor where M is the number of SDfs in the scene. Each 
                   element of idxs is an index of an SDF in the sdfs list passed
                   to the constructor.
            -poses: A BxMx(D+R) tensor with the pose of each of the M tensors in 
                    idxs in each row. Each row is D+R in length, where the first
                    D elements are the translation and the remaining R elements
                    are the rotation. R will vary in size and format depending on
                    D as follows:
                        -D=1: R is 0 (no rotation in 1D)
                        -D=2: R is 1 and is the angle in radians, where 0 is +x 
                              and pi/2 is +y.
                        -D=3: R is 4 and is the quaternion representation of the
                              3D rotation in xyzw format and normalized.
                    All other values for D are not supported.
            -scales: A BxM tensor with the scale

        Returns: A BxNxO tensor where O is the number of output features.
        """

        # Error checking.
        batch_size = locs.size()[0]
        N = locs.size()[1]
        M = idxs.size()[1]
        R = {1: 0, 2: 1, 3: 4}[self.ndim]
        ec.check_tensor_dims(locs, "locs", (batch_size, N, self.ndim))
        ec.check_tensor_dims(idxs, "idxs", (batch_size, M,))
        ec.check_tensor_dims(poses, "poses", (batch_size, M, self.ndim + R))
        ec.check_tensor_dims(scales, "scales", (batch_size, M,))

        locs = locs.contiguous()
        idxs = idxs.contiguous()
        poses = poses.contiguous()
        scales = scales.contiguous()

        # Do the compution.
        convsdf = _ConvSDFFunction(self.sdfs, self.sdf_offsets, self.sdf_shapes,
                                   self.kernel_size, self.dilation, self.max_distance, self.compute_pose_grads)
        return convsdf(locs, idxs, poses, scales, self.weight, self.bias)


"""

INTERNAL FUNCTIONS

"""


class _ConvSDFFunction(torch.autograd.Function):

    def __init__(self, sdfs, sdf_offsets, sdf_shapes, kernel_size, dilation, max_distance, compute_pose_grads):
        super(_ConvSDFFunction, self).__init__()
        self.sdfs = sdfs
        self.sdf_offsets = sdf_offsets
        self.sdf_shapes = sdf_shapes
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.max_distance = max_distance
        self.compute_pose_grads = compute_pose_grads

    def forward(self, locs, idxs, poses, scales, weight, bias):
        self.save_for_backward(locs, idxs, poses, scales, weight, bias)
        batch_size = locs.size()[0]
        N = locs.size()[1]
        nkernels = weight.size()[0]
        ret = self.sdfs.new(batch_size, N, nkernels)
        ret.fill_(0)
        if locs.is_cuda:
            if not _extc.spnc_convsdf_forward(locs, idxs, poses, scales, self.sdfs, self.sdf_offsets,
                                              self.sdf_shapes, weight, bias, self.kernel_size, self.dilation,
                                              self.max_distance, ret):
                raise Exception("Cuda error")
        else:
            _ext.spn_convsdf_forward(locs, idxs, poses, scales, self.sdfs, self.sdf_offsets,
                                     self.sdf_shapes, weight, bias, self.kernel_size, self.dilation,
                                     self.max_distance, ret)

        return ret

    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        locs, idxs, poses, scales, weight, bias = self.saved_tensors
        ret_locs = grad_output.new(locs.size())
        ret_locs.fill_(0)
        ret_weight = grad_output.new(weight.size())
        ret_weight.fill_(0)
        if self.compute_pose_grads:
            ret_poses = grad_output.new(poses.size())
        else:
            ret_poses = grad_output.new(poses.size()[0] + 1)
        ret_poses.fill_(0)
        if grad_output.is_cuda:
            if not _extc.spnc_convsdf_backward(locs, idxs, poses, scales, self.sdfs, self.sdf_offsets,
                                               self.sdf_shapes, weight, bias, self.kernel_size, self.dilation,
                                               self.max_distance, grad_output, ret_locs, ret_weight, ret_poses):
                raise Exception("Cuda error")
        else:
            _ext.spn_convsdf_backward(locs, idxs, poses, scales, self.sdfs, self.sdf_offsets,
                                      self.sdf_shapes, weight, bias, self.kernel_size, self.dilation,
                                      self.max_distance, grad_output, ret_locs, ret_weight, ret_poses)

        if not self.compute_pose_grads:
            ret_poses = grad_output.new(poses.size())
            ret_poses.fill_(0)

        # TODO: There's a bug in the analytical gradients wrt rotations. Use numerical for now.
        # I cannot for the life of me get gradients through qauternion rotation right.
        if self.compute_pose_grads:
            # ret_poses = grad_output.new(poses.size())
            # ret_poses.fill_(0)
            baseline = self.forward(locs, idxs, poses, scales, weight, bias)
            for m in range(poses.size()[1]):
                for i in range(locs.size()[-1], poses.size()[2]):
                    poses[:, m, i] += 1e-3
                    nn = self.forward(locs, idxs, poses, scales, weight, bias)
                    poses[:, m, i] -= 1e-3
                    gg = (nn - baseline)/1e-3
                    ret_poses[:, m, i] = torch.sum(
                        torch.sum(gg*grad_output, 1), 1)

        # PyTorch requires gradients for each input, but we only care about the
        # gradients for weight and bias, so set the rest to 0.
        return (ret_locs,
                grad_output.new(idxs.size()).fill_(0),
                ret_poses,
                grad_output.new(scales.size()).fill_(0),
                ret_weight,
                grad_output.sum(1).sum(0))
