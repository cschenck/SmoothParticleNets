import os
import sys
# Add path to python source to path.
sys.path.append(os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "python"))
import SmoothParticleNets as spn

import itertools
import numpy as np
import torch
import torch.autograd

from gradcheck import gradcheck
from test_convsdf import quaternionMult, quaternionConjugate
from regular_grid_interpolater import RegularGridInterpolator
try:
    import pytest_args
except ImportError:
    print("Make sure to compile SmoothParticleNets before running tests.")
    raise


def pyproject(locs, image, camera_fl, camera_pose,
              camera_rot, depth_mask=None, dtype=np.float32):
    batch_size = locs.shape[0]
    N = locs.shape[1]
    channels = image.shape[1]
    width = image.shape[3]
    height = image.shape[2]
    ret = np.zeros((batch_size, N, channels), dtype=dtype)
    if depth_mask is None:
        depth_mask = np.ones((batch_size, height, width),
                             dtype=dtype)*np.finfo(np.float32).max
    depth_fns = [RegularGridInterpolator(
        [np.arange(0.5, width, 1), np.arange(0.5, height, 1)],
        depth_mask[b, ...].transpose(), bounds_error=False, fill_value=np.finfo(np.float32).max)
        for b in range(batch_size)]
    for b in range(batch_size):
        r = locs[b, ...] - camera_pose[b, ...]
        r = np.concatenate((r, np.zeros((N, 1), dtype=r.dtype)), axis=-1)
        r = np.array([quaternionMult(quaternionConjugate(camera_rot[b, :]),
                                     quaternionMult(r[i, ...], camera_rot[b, :])) for i in range(N)], dtype=dtype)
        ijs = np.concatenate((
            r[:, 0:1]*camera_fl/r[:, 2:3] + width/2.0,
            r[:, 1:2]*camera_fl/r[:, 2:3] + height/2.0,
        ), axis=-1)
        depths = depth_fns[b](ijs)
        mask = (r[:, 2] <= depths)*(r[:, 2] > 0)
        for c in range(channels):
            fn = RegularGridInterpolator(
                [np.arange(0.5, width, 1), np.arange(0.5, height, 1)],
                image[b, c, ...].transpose(), bounds_error=False, fill_value=0)
            ret[b, :, c] = fn(ijs)*mask

    return ret


def test_imageprojection(cpu=True, cuda=True):
    if cpu:
        print("Testing CPU implementation of ImageProjection...")
        eval_imageprojection(cuda=False)
        print("CPU implementation passed!")
        print("")

    if cuda:
        if pytest_args.with_cuda:
            print("Testing CUDA implementation of ImageProjection...")
            eval_imageprojection(cuda=True)
            print("CUDA implementation passed!")
        else:
            print("Not compiled with CUDA, skipping CUDA test.")


def eval_imageprojection(cuda=False):
    np.random.seed(1)
    BATCH_SIZE = 2
    N = 5
    CHANNELS = 2
    CAMERA_FOV = 45.0/180.0*np.pi
    CAMERA_SIZE = (30, 30)
    CAMERA_FL = CAMERA_SIZE[0]/2/(CAMERA_FOV/2.0)
    CAMERA_POSE = 5.0*(np.random.rand(BATCH_SIZE, 3).astype(np.float32) - 0.5)
    CAMERA_TARGET = np.array([(0.0, 0.0, 0.0)]*BATCH_SIZE, dtype=np.float32)

    CAMERA_ROT = np.zeros((BATCH_SIZE, 4), dtype=np.float32)
    for b in range(BATCH_SIZE):
        CAMERA_ROT[b, :] = pointAt(
            CAMERA_POSE[b, :], np.array([0, 0, 0], dtype=np.float32))

    locs = 2.0*(np.random.rand(BATCH_SIZE, N, 3).astype(np.float32) - 0.5)
    image = np.random.rand(BATCH_SIZE, CHANNELS,
                           CAMERA_SIZE[1], CAMERA_SIZE[0])
    depth_mask = np.ones((BATCH_SIZE, CAMERA_SIZE[1], CAMERA_SIZE[0]),
                         dtype=np.float32)*np.finfo(np.float32).max
    ir = (int(CAMERA_SIZE[0]/2 - CAMERA_SIZE[0]*0.2),
          int(CAMERA_SIZE[0]/2 + CAMERA_SIZE[0]*0.2) + 1)
    jr = (int(CAMERA_SIZE[1]/2 - CAMERA_SIZE[1]*0.2),
          int(CAMERA_SIZE[1]/2 + CAMERA_SIZE[1]*0.2) + 1)
    ul = 0.0
    lr = 10.0
    ur = 5.0
    ll = 3.5
    for i in range(ir[0], ir[1]):
        for j in range(jr[0], jr[1]):
            ii = 1.0*(i - ir[0])/(ir[1] - ir[0])
            jj = 1.0*(j - jr[0])/(jr[1] - jr[0])
            l = ul*(1 - jj) + ll*jj
            r = ur*(1 - jj) + lr*jj
            depth_mask[0, j, i] = l*(1 - ii) + r*ii

    def use_cuda(x):
        if cuda:
            return x.cuda()
        else:
            return x

    def undo_cuda(x):
        if cuda:
            return x.cpu()
        else:
            return x

    def np2var(t):
        return torch.autograd.Variable(use_cuda(torch.from_numpy(t)), requires_grad=False)

    locs_t = torch.autograd.Variable(
        use_cuda(torch.FloatTensor(locs)), requires_grad=True)
    image_t = torch.autograd.Variable(
        use_cuda(torch.FloatTensor(image)), requires_grad=True)
    depth_mask_t = torch.autograd.Variable(
        use_cuda(torch.FloatTensor(depth_mask)), requires_grad=False)
    camera_pose_t = torch.autograd.Variable(use_cuda(torch.FloatTensor(CAMERA_POSE)),
                                            requires_grad=False)
    camera_rot_t = torch.autograd.Variable(use_cuda(torch.FloatTensor(CAMERA_ROT)),
                                           requires_grad=False)

    imageProjection = spn.ImageProjection(CAMERA_FL)

    ground_truth = pyproject(locs, image, CAMERA_FL,
                             CAMERA_POSE, CAMERA_ROT, depth_mask)
    pred_t = imageProjection(
        locs_t, image_t, camera_pose_t, camera_rot_t, depth_mask_t)
    pred = undo_cuda(pred_t).data.numpy()
    np.testing.assert_array_almost_equal(pred, ground_truth, decimal=3)

    # Use pyproject to allow for double precision when computing numeric grads.
    def func_numerical(l, i):
        ll = undo_cuda(l).data.numpy()
        ii = undo_cuda(i).data.numpy()
        return torch.autograd.Variable(use_cuda(torch.from_numpy(pyproject(ll, ii, CAMERA_FL, CAMERA_POSE,
                                                                           CAMERA_ROT, dtype=np.float64))), requires_grad=False)

    def func_analytical(l, i):
        return imageProjection(l, i, camera_pose_t, camera_rot_t)
    assert torch.autograd.gradcheck(func_analytical, (locs_t, image_t,),
                                    eps=1e-3, atol=1e-3, rtol=1e-1)


def quaternionFromMatrix(matrix):
    M = matrix
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                  [m01+m10,     m11-m00-m22, 0.0,         0.0],
                  [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                  [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
    K /= 3.0
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return [q[1], q[2], q[3], q[0]]


def pointAt(pose, target):
    # Convention: +Z=out of camera, +Y=Down, +X=right
    z = target - pose
    z /= np.sqrt(np.sum(z**2))
    y = np.array([0, -1, 0], dtype=np.float32)
    x = np.cross(y, z)
    x /= np.sqrt(np.sum(x**2))
    y = np.cross(z, x)
    ret = quaternionFromMatrix(np.array([x, y, z]).transpose())
    return ret


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', dest='cpu', action="store_true", default=True)
    parser.add_argument('--no-cpu', dest='cpu', action="store_false")
    parser.add_argument('--cuda', dest='cuda',
                        action="store_true", default=True)
    parser.add_argument('--no-cuda', dest='cuda', action="store_false")
    args = parser.parse_args()
    test_imageprojection(cpu=args.cpu, cuda=args.cuda)
