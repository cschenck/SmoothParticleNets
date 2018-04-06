

import itertools
import os
import sys
# Add path to python source to path.
sys.path.append(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import SmoothParticleNets as spn
import math
import numpy as np
import Queue
import torch
import torch.autograd

from gradcheck import gradcheck
from regular_grid_interpolater import RegularGridInterpolator
try:
    import pytest_args
except ImportError:
    print("Make sure to compile SmoothParticleNets before running tests.")
    raise

def construct_sdf(sdf_shape, nugget, width):
    scale = [1.0*width[i]/sdf_shape[i] for i in range(len(width))]
    ret = np.ones(sdf_shape, dtype=np.float32)*np.finfo(np.float32).max
    ret[nugget] = 0
    return prop_sdf(ret, [(0, nugget)], scale=scale)

def prop_sdf(sdf, initial_frontier, scale=[1, 1, 1]):
    sdf_shape = sdf.shape
    closed = set()
    frontier = Queue.PriorityQueue()
    for c, p in initial_frontier:
        frontier.put((c, tuple(p)))
    while not frontier.empty():
        cost, current = frontier.get()
        if current in closed:
            continue
        else:
            closed.add(current)
        for i in [-1, 0, 1]:
            if current[0] + i < 0 or current[0] + i >= sdf_shape[0]:
                continue
            for j in [-1, 0, 1]:
                if current[1] + j < 0 or current[1] + j >= sdf_shape[1]:
                    continue
                for k in [-1, 0, 1]:
                    if current[2] + k < 0 or current[2] + k >= sdf_shape[2]:
                        continue
                    if j == 0 and i == 0 and k == 0:
                        continue
                    nn = (current[0] + i, current[1] + j, current[2] + k)
                    d = np.sqrt(np.sum(np.square([i*scale[0], j*scale[1], k*scale[2]])))
                    if  cost + d < sdf[nn]:
                        sdf[nn] = cost + d
                        frontier.put((cost + d, nn))
    return sdf


def test_convsdf():
    print("Testing CPU implementation of ConvSDF...")
    eval_convsdf(cuda=False)
    print("CPU implementation passed!")
    print("")

    if pytest_args.with_cuda:
        print("Testing CUDA implementation of ConvSDF...")
        eval_convsdf(cuda=True)
        print("CUDA implementation passed!")
    else:
        print("Not compiled with CUDA, skipping CUDA test.")

def eval_convsdf(cuda=False):
    BATCH_SIZE = 2
    N = 10
    M = 30
    NDIM = 3
    KERNEL_SIZE = (3, 5, 3)
    DILATION = 0.05
    NKERNELS = 2
    MAX_DISTANCE = 13.37

    np.random.seed(0)

    locs = np.random.rand(BATCH_SIZE, N, NDIM)
    weights = np.random.rand(NKERNELS, np.prod(KERNEL_SIZE))
    biases = np.random.rand(NKERNELS)

    kernel_centers = (np.array(KERNEL_SIZE) - 1)/2
    ground_truth = np.zeros((BATCH_SIZE, N, NKERNELS), dtype=np.float32)

    sdf_widths = np.array([[4, 4, 4],
                           [6, 4, 8],
                           [5, 5, 5],
                          ], dtype=np.float32)/1
    sdf1 = construct_sdf((5, 5, 5), (1, 2, 3), sdf_widths[0, :])
    sdf2 = construct_sdf((6, 4, 8), (1, 0, 2), sdf_widths[1, :])
    sdf3 = construct_sdf((20, 20, 20), (5, 15, 5), sdf_widths[2, :])
    sdfs = [sdf1, sdf2, sdf3]

    sdf_poses = np.random.rand(BATCH_SIZE, M, 7)
    # Convert axis angle to quaternion
    sdf_poses[..., 3:-1] *= np.sin(sdf_poses[..., -1, np.newaxis]/2)
    sdf_poses[..., -1] = np.cos(sdf_poses[..., -1]/2)
    sdf_poses[..., 3:] /= np.sqrt((sdf_poses[..., 3:]**2).sum(axis=-1))[..., np.newaxis]

    idxs = np.random.randint(0, 3, size=(BATCH_SIZE, M))
    idxs[-1, -1] = -1
    scales = np.random.rand(BATCH_SIZE, M)*0.5 + 0.5

    sdf_fns = [RegularGridInterpolator(
                    [np.linspace(0.5, y - 0.5, y)*s/y
                        for y, s in zip(sdfs[i].shape, sdf_widths[i, ...])], sdfs[i], 
                    bounds_error=False, fill_value=np.finfo(np.float32).max) 
                for i in range(len(sdfs))]

    for outk in range(NKERNELS):
        allkidx = itertools.product(
            *[list(range(-(k//2), k//2 + 1)) for k in KERNEL_SIZE[::-1]])
        for k, kidx in enumerate(allkidx):
            for i in range(N):
                for b in range(BATCH_SIZE):
                    r = locs[b, i, :] + [kidx[::-1][j]*DILATION for j in range(3)]
                    minv = MAX_DISTANCE
                    for m in range(M):
                        mm = idxs[b, m]
                        if mm < 0:
                            continue
                        r2 = quaternionMult(quaternionConjugate(sdf_poses[b, m, 3:]), 
                                            quaternionMult(r - sdf_poses[b, m, :3], 
                                                           sdf_poses[b, m, 3:]))[:3]
                        r2 /= scales[b, m]
                        v = sdf_fns[mm](r2)*scales[b, m]
                        minv = min(v, minv)
                    ground_truth[b, i, outk] += weights[outk, k]*minv
    ground_truth += biases[np.newaxis, np.newaxis, :]

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

    sdfs_t = [torch.FloatTensor(x) for x in sdfs]
    sdf_sizes_t = [np.mean([1.0*sdf_widths[i, j]/sdfs[i].shape[j] 
                    for j in range(len(sdfs[i].shape))]) for i in range(len(sdfs))] 
    locs_t = torch.autograd.Variable(use_cuda(torch.FloatTensor(locs)), requires_grad=True)
    idxs_t = torch.autograd.Variable(use_cuda(torch.FloatTensor(idxs)), requires_grad=False)
    poses_t = torch.autograd.Variable(use_cuda(torch.FloatTensor(sdf_poses)), 
                                        requires_grad=False)
    scales_t = torch.autograd.Variable(use_cuda(torch.FloatTensor(scales)), 
                                        requires_grad=False)
    weights_t = torch.nn.Parameter(torch.FloatTensor(weights), requires_grad=True)
    biases_t = torch.nn.Parameter(torch.FloatTensor(biases), requires_grad=True)

    convsdf = spn.ConvSDF(sdfs_t, sdf_sizes_t, NKERNELS, NDIM, KERNEL_SIZE, DILATION, 
                            MAX_DISTANCE)
    convsdf.weight = weights_t
    convsdf.bias = biases_t
    convsdf = use_cuda(convsdf)

    pred = undo_cuda(convsdf(locs_t, idxs_t, poses_t, scales_t))

    np.testing.assert_array_almost_equal(pred.data.numpy(), ground_truth, decimal=3)

    def func(w, b):
        convsdf.weight = w
        convsdf.bias = b
        return (convsdf(locs_t, idxs_t, poses_t, scales_t),)
    assert gradcheck(func, (weights_t, biases_t), eps=1e-2, atol=1e-3)
    
    
def quaternionMult(q1, q2):
    if len(q1) == 4:
        x1, y1, z1, w1 = q1
    else:
        x1, y1, z1 = q1
        w1 = 0
    if len(q2) == 4:
        x2, y2, z2, w2 = q2
    else:
        x2, y2, z2 = q2
        w2 = 0
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([x, y, z, w])
    
def quaternionConjugate(q):
    x, y, z, w = q
    return [-x, -y, -z, w]
    

if __name__ == '__main__':
    test_convsdf()