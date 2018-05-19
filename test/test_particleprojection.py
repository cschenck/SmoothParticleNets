import os
import sys
# Add path to python source to path.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import SmoothParticleNets as spn

import itertools
import numpy as np
import torch
import torch.autograd

from gradcheck import gradcheck
from test_convsdf import quaternionMult, quaternionConjugate
try:
    import pytest_args
except ImportError:
    print("Make sure to compile SmoothParticleNets before running tests.")
    raise


def pyproject(camera_fl, camera_size, filter_std, filter_scale, locs, camera_pose, 
    camera_rot, depth_mask=None, dtype=np.float32):
    batch_size = locs.shape[0]
    N = locs.shape[1]
    ret = np.zeros((batch_size, camera_size[1], camera_size[0]), dtype=dtype)
    if depth_mask is None:
        depth_mask = np.ones((batch_size, camera_size[1], camera_size[0]), 
            dtype=dtype)*np.finfo(np.float32).max
    for b in range(batch_size):
        for n in range(N):
            r = locs[b, n, :]
            t = r - camera_pose[b, ...]
            t = np.array([t[0], t[1], t[2], 0.0], dtype=dtype)
            t = np.array(quaternionMult(quaternionConjugate(camera_rot[b, :]), 
                    quaternionMult(t, camera_rot[b, :])), dtype=dtype)
            if t[2] <= 0:
                continue
            p = np.array([t[0]*camera_fl/t[2] + camera_size[0]/2.0, 
                          t[1]*camera_fl/t[2] + camera_size[1]/2.0], 
                dtype=dtype)
            s = np.ceil(filter_std*2)
            f = filter_scale/(filter_std*np.sqrt(2*np.pi))
            for i in np.arange(max(0, p[0] - s), min(camera_size[0], p[0] + s + 1), 1):
                for j in np.arange(max(0, p[1] - s), min(camera_size[1], p[1] + s + 1), 1):
                    if depth_mask[b, int(j), int(i)] < t[2]:
                        continue
                    xi = int(i) + 0.5
                    yj = int(j) + 0.5
                    d2 = (xi - p[0])**2 + (yj - p[1])**2
                    if d2 > s*s:
                        continue
                    v = f*np.exp(-d2/(2.0*filter_std*filter_std))
                    ret[b, int(j), int(i)] += v
    return ret




def test_particleprojection(cpu=True, cuda=True):
    if cpu:
        print("Testing CPU implementation of ParticleProjection...")
        eval_particleprojection(cuda=False)
        print("CPU implementation passed!")
        print("")

    if cuda:
        if pytest_args.with_cuda:
            print("Testing CUDA implementation of ParticleProjection...")
            eval_particleprojection(cuda=True)
            print("CUDA implementation passed!")
        else:
            print("Not compiled with CUDA, skipping CUDA test.")

def eval_particleprojection(cuda=False):
    np.random.seed(1)
    BATCH_SIZE = 2
    N = 5
    CAMERA_FOV = 45.0/180.0*np.pi
    CAMERA_SIZE = (120, 90)
    # CAMERA_SIZE = (1024, 768)
    CAMERA_FL = CAMERA_SIZE[0]/2/(CAMERA_FOV/2.0)
    FILTER_STD = 5
    FILTER_SCALE = 1.0/0.06
    CAMERA_POSE = 5.0*(np.random.rand(BATCH_SIZE, 3).astype(np.float32) - 0.5)
    CAMERA_TARGET = np.array([(0.0, 0.0, 0.0)]*BATCH_SIZE, dtype=np.float32)

    CAMERA_ROT = np.zeros((BATCH_SIZE, 4), dtype=np.float32)
    for b in range(BATCH_SIZE):
        CAMERA_ROT[b, :] = pointAt(CAMERA_POSE[b, :], np.array([0, 0, 0], dtype=np.float32))

    locs = 2.0*(np.random.rand(BATCH_SIZE, N, 3).astype(np.float32) - 0.5)
    depth_mask = np.ones((BATCH_SIZE, CAMERA_SIZE[1], CAMERA_SIZE[0]), 
        dtype=np.float32)*np.finfo(np.float32).max
    ir = (int(CAMERA_SIZE[0]/2 - CAMERA_SIZE[0]*0.2), int(CAMERA_SIZE[0]/2 + CAMERA_SIZE[0]*0.2) + 1)
    jr = (int(CAMERA_SIZE[1]/2 - CAMERA_SIZE[1]*0.2), int(CAMERA_SIZE[1]/2 + CAMERA_SIZE[1]*0.2) + 1)
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


    locs_t = torch.autograd.Variable(use_cuda(torch.FloatTensor(locs)), requires_grad=True)
    depth_mask_t = torch.autograd.Variable(use_cuda(torch.FloatTensor(depth_mask)), requires_grad=False)
    camera_pose_t = torch.autograd.Variable(use_cuda(torch.FloatTensor(CAMERA_POSE)), 
        requires_grad=False)
    camera_rot_t = torch.autograd.Variable(use_cuda(torch.FloatTensor(CAMERA_ROT)), 
        requires_grad=False)

    particleProjection = spn.ParticleProjection(CAMERA_FL, CAMERA_SIZE, FILTER_STD, FILTER_SCALE)

    # particleViewer([
    #         lambda p, r: pyproject(CAMERA_FL, CAMERA_SIZE, FILTER_STD, FILTER_SCALE, locs, 
    #                         p, r, depth_mask),
    #         lambda p, r: undo_cuda(particleProjection(locs_t, np2var(p), np2var(r), 
    #                         depth_mask_t)).data.numpy(),
    #     ], BATCH_SIZE, 5, ["Ground Truth", "Output"])
    # return

    ground_truth = pyproject(CAMERA_FL, CAMERA_SIZE, FILTER_STD, FILTER_SCALE, locs, 
        CAMERA_POSE, CAMERA_ROT, depth_mask)
    pred_t = particleProjection(locs_t, camera_pose_t, camera_rot_t, depth_mask_t)
    pred = undo_cuda(pred_t).data.numpy()
    # visualizeOutput([ground_truth, pred, -(pred - ground_truth)], 
    #     ["Ground Truth", "Prediction", "Difference"])
    np.testing.assert_array_almost_equal(pred, ground_truth, decimal=3)

    # Use pyconvsp to allow for double precision when computing numeric grads.
    def func_numerical(l):
        ll = undo_cuda(l).data.numpy()
        return torch.autograd.Variable(use_cuda(torch.from_numpy(pyproject(CAMERA_FL, CAMERA_SIZE, 
            FILTER_STD, FILTER_SCALE, ll, CAMERA_POSE, CAMERA_ROT, dtype=np.float64))), 
            requires_grad=False)
    def func_analytical(l):
        return particleProjection(l, camera_pose_t, camera_rot_t)
    assert gradcheck(func_analytical, (locs_t,), eps=1e-6, atol=1e-3, rtol=1e-2,
        func_numerical=func_numerical, use_double=True)




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
    K =    np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
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


def visualizeOutput(outputs, titles=None):
    import cv2
    if titles is None:
        titles = [str(i) for i in range(len(funcs))]
    for ret, title in zip(outputs, titles):
        img = np.ones(((ret.shape[1] + 5)*ret.shape[0], ret.shape[2]), dtype=np.float32)
        for b in range(ret.shape[0]):
            i = b*(ret.shape[1] + 5)
            j = (b + 1)*(ret.shape[1] + 5) - 5
            img[i:j, :] = ret[b, ...]
        cv2.imshow(title, img)
    return cv2.waitKey(0)

# Utility function for visualizing particles.
def particleViewer(funcs, batch_size, radius, titles=None):
    ESCAPE = 1048603
    LEFT = 1113937
    RIGHT = 1113939
    UP = 1113938
    DOWN = 1113940
    hangle = 0.0
    vangle = 0.0
    if titles is None:
        titles = [str(i) for i in range(len(funcs))]
    k = None
    while k != ESCAPE:
        y = radius*np.sin(vangle)
        r = radius*np.cos(vangle)
        x = r*np.cos(hangle)
        z = r*np.sin(hangle)
        pose = [x, y, z]
        camera_pose = np.array([pose]*batch_size, dtype=np.float32)
        camera_rot = np.array([pointAt(pose, np.array([0, 0, 0], dtype=np.float32))]*batch_size, 
            dtype=np.float32)
        k = visualizeOutput([func(camera_pose, camera_rot) for func in funcs], titles)
        if k == LEFT:
            hangle += 1.0/180.0*np.pi
        if k == RIGHT:
            hangle -= 1.0/180.0*np.pi
        if k == UP:
            vangle += 1.0/180.0*np.pi
        if k == DOWN:
            vangle -= 1.0/180.0*np.pi
        if vangle > np.pi/2:
            vangle = np.pi/2
        elif vangle < -np.pi/2:
            vangle = -np.pi/2



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', dest='cpu', action="store_true", default=True)
    parser.add_argument('--no-cpu', dest='cpu', action="store_false")
    parser.add_argument('--cuda', dest='cuda', action="store_true", default=True)
    parser.add_argument('--no-cuda', dest='cuda', action="store_false")
    args = parser.parse_args()
    test_particleprojection(cpu=args.cpu, cuda=args.cuda)