
import numbers
import numpy as np

import torch
import torch.autograd

import _ext
import error_checking as ec
from kernels import KERNELS, KERNEL_NAMES

MAX_FLOAT = float(np.finfo(np.float32).max)

class ParticleProjection(torch.nn.Module):
    """ 
    """
    def __init__(self, camera_fl, camera_size, filter_std, filter_scale):
        """ 
        """
        super(ParticleProjection, self).__init__()

        self.camera_size = ec.make_list(camera_size, 2, "camera_size", 
            "%s > 0", "isinstance(%s, numbers.Integral)")

        self.camera_fl = ec.check_conditions(camera_fl, "camera_fl", 
            "%s > 0", "isinstance(%s, numbers.Real)")
        self.filter_std = ec.check_conditions(filter_std, "filter_std", 
            "%s > 0", "isinstance(%s, numbers.Real)")
        self.filter_scale = ec.check_conditions(filter_scale, "filter_scale", 
            "%s > 0", "isinstance(%s, numbers.Real)")

        self.register_buffer("empty_depth_mask", 
            torch.ones(1, self.camera_size[1], self.camera_size[0])*MAX_FLOAT)

    def _rotationMatrixFromQuaternion(self, quat):
        """
        1 - 2*qy2 - 2*qz2   2*qx*qy - 2*qz*qw   2*qx*qz + 2*qy*qw
        2*qx*qy + 2*qz*qw   1 - 2*qx2 - 2*qz2   2*qy*qz - 2*qx*qw
        2*qx*qz - 2*qy*qw   2*qy*qz + 2*qx*qw   1 - 2*qx2 - 2*qy2
        """
        quat = quat.data
        qx = quat[:, 0]
        qy = quat[:, 1]
        qz = quat[:, 2]
        qw = quat[:, 3]
        qx2 = qx*qx
        qxqy = qx*qy
        qxqz = qx*qz
        qxqw = qx*qw
        qy2 = qy*qy
        qyqz = qy*qz
        qyqw = qy*qw
        qz2 = qz*qz
        qzqw = qz*qw
        ret = quat.new(quat.size()[0], 3, 3)
        ret[:, 0, 0] = 1 - 2*qy2 - 2*qz2
        ret[:, 1, 0] = 2*qxqy - 2*qzqw
        ret[:, 2, 0] = 2*qxqz + 2*qyqw
        ret[:, 0, 1] = 2*qxqy + 2*qzqw
        ret[:, 1, 1] = 1 - 2*qx2 - 2*qz2
        ret[:, 2, 1] = 2*qyqz - 2*qxqw
        ret[:, 0, 2] = 2*qxqz - 2*qyqw
        ret[:, 1, 2] = 2*qyqz + 2*qxqw
        ret[:, 2, 2] = 1 - 2*qx2 - 2*qy2
        return torch.autograd.Variable(ret, requires_grad=False)

    def forward(self, locs, camera_pose, camera_rot, depth_mask=None):
        """ 
        """

        # Error checking.
        batch_size = locs.size()[0]
        N = locs.size()[1]
        ec.check_tensor_dims(locs, "locs", (batch_size, N, 3))
        ec.check_tensor_dims(camera_pose, "camera_pose", (batch_size, 3))
        ec.check_tensor_dims(camera_rot, "camera_rot", (batch_size, 4))

        if depth_mask is not None:
            ec.check_tensor_dims(depth_mask, "depth_mask", (batch_size, 
                self.camera_size[1], self.camera_size[0]))
            depth_mask = depth_mask.contiguous()
        else:
            if self.empty_depth_mask.size()[0] != batch_size:
                self.empty_depth_mask.resize_(batch_size, self.camera_size[1], self.camera_size[0])
                self.empty_depth_mask.fill_(MAX_FLOAT)
            depth_mask = torch.autograd.Variable(self.empty_depth_mask, requires_grad=False)
            if locs.is_cuda:
                depth_mask = depth_mask.cuda()

        # Let's transform the particles to camera space here.
        locs = locs - camera_pose.unsqueeze(1)
        # Ensure the rotation quaternion is normalized.
        camera_rot = camera_rot/torch.sqrt(torch.sum(camera_rot**2, 1, keepdim=True))
        # Invert the rotation.
        inv = camera_rot.data.new(1, 4)
        inv[0, 0] = -1
        inv[0, 1] = -1
        inv[0, 2] = -1
        inv[0, 3] = 1
        inv = torch.autograd.Variable(inv, requires_grad=False)
        camera_rot = camera_rot*inv
        rot = self._rotationMatrixFromQuaternion(camera_rot)
        # Rotate the locs into camera space.
        locs = torch.bmm(locs, rot)

        locs = locs.contiguous()
        proj = _ParticleProjectionFunction(self.camera_fl, self.camera_size, self.filter_std,
            self.filter_scale)
        ret = proj(locs, depth_mask)
        return ret
        



"""

INTERNAL FUNCTIONS

"""

class _ParticleProjectionFunction(torch.autograd.Function):

    def __init__(self, camera_fl, camera_size, filter_std, filter_scale):
        super(_ParticleProjectionFunction, self).__init__()
        self.camera_fl = camera_fl
        self.camera_size = camera_size
        self.filter_std = filter_std
        self.filter_scale = filter_scale

    def forward(self, locs, depth_mask):
        self.save_for_backward(locs, depth_mask)
        batch_size = locs.size()[0]
        ret = locs.new(batch_size, self.camera_size[1], self.camera_size[0])
        ret.fill_(0)
        if locs.is_cuda:
            if not _ext.spnc_particleprojection_forward(locs, self.camera_fl,
                self.filter_std, self.filter_scale, depth_mask, ret):
                raise Exception("Cuda error")
        else:
            _ext.spn_particleprojection_forward(locs, self.camera_fl,
                self.filter_std, self.filter_scale, depth_mask, ret)

        return ret 


    def backward(self, grad_output):
        locs, depth_mask = self.saved_tensors
        ret_locs = grad_output.new(locs.size())
        ret_locs.fill_(0)
        ret_depth_mask = grad_output.new(depth_mask.size())
        ret_depth_mask.fill_(0)
        if grad_output.is_cuda:
            if not _ext.spnc_particleprojection_backward(locs,  
                self.camera_fl, self.filter_std, self.filter_scale, depth_mask, grad_output, ret_locs):
                raise Exception("Cuda error")
        else:
            _ext.spn_particleprojection_backward(locs, 
                self.camera_fl, self.filter_std, self.filter_scale, depth_mask, grad_output, ret_locs)

        return (ret_locs,
                ret_depth_mask,)




