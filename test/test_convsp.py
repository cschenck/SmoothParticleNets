    import os
import sys
# Add path to python source to path.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import SmoothParticleNets as spn

import itertools
import numpy as np
import torch
import torch.autograd

try:
    import pytest_args
except ImportError:
    print("Make sure to compile SmoothParticleNets before running tests.")
    raise

# The kernel function for SPH.
def w(x, h=1):
    return 1.0/np.pi*(0.25*max(0, h - x)**3 - max(0, h/2.0 - x)**3)/(h**3/(8*np.pi))

def test_convsp():
    print("Testing CPU implementation of ConvSP...")
    eval_convsp(cuda=False)
    print("CPU implementation passed!")
    print("")

    if pytest_args.with_cuda:
        print("Testing CUDA implementation of ConvSP...")
        eval_convsp(cuda=True)
        print("CUDA implementation passed!")
    else:
        print("Not compiled with CUDA, skipping CUDA test.")

def eval_convsp(cuda=False):
    BATCH_SIZE = 2
    N = 10
    NDIM = 2
    KERNEL_SIZE = (3, 5)
    RADIUS = 1.0
    DILATION = 0.05
    NCHANNELS = 2
    NKERNELS = 3
    MASS = 1.0

    np.random.seed(0)

    locs = np.random.rand(BATCH_SIZE, N, NDIM + 1)
    locs[..., -1] = 1.0/MASS
    data = np.random.rand(BATCH_SIZE, NCHANNELS, N)
    density = np.zeros((BATCH_SIZE, N), dtype=np.float32)
    for b in range(BATCH_SIZE):
        for i in range(N):
            for j in range(N):
                d = np.square(locs[b, i, :NDIM] - locs[b, j, :NDIM]).sum()
                if d < RADIUS*RADIUS:
                    d = w(np.sqrt(d), h=RADIUS)
                    density[b, i] += d/locs[b, j, -1]
    weights = np.random.rand(NKERNELS, NCHANNELS, np.prod(KERNEL_SIZE))
    biases = np.random.rand(NKERNELS)

    kernel_centers = (np.array(KERNEL_SIZE) - 1)/2
    ground_truth = np.zeros((BATCH_SIZE, NKERNELS, N), dtype=np.float32)
    for b in range(BATCH_SIZE):
        for i in range(N):
            for j in range(N):
                d = np.square(locs[b, i, :NDIM] - locs[b, j, :NDIM]).sum()
                nr = DILATION*max(KERNEL_SIZE)/2 + RADIUS
                if d > nr*nr:
                    continue
                for k, idxs in enumerate(itertools.product(*[range(x) for x in KERNEL_SIZE[::-1]])):
                    d = np.square(locs[b, i, :NDIM] + (idxs[::-1] - kernel_centers)*DILATION 
                        - locs[b, j, :NDIM]).sum()
                    if d > RADIUS*RADIUS:
                        continue
                    ground_truth[b, :, i] += weights[:, :, k].dot(
                        1.0/(locs[b, j, NDIM]*density[b, j])
                        *w(np.sqrt(d), h=RADIUS)*data[b, :, j])
    ground_truth += biases[np.newaxis, :, np.newaxis]

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

    locs = torch.autograd.Variable(use_cuda(torch.FloatTensor(locs)), requires_grad=False)
    data = torch.autograd.Variable(use_cuda(torch.FloatTensor(data)), requires_grad=True)
    density = torch.autograd.Variable(use_cuda(torch.FloatTensor(density)), requires_grad=False)
    weights = torch.nn.Parameter(use_cuda(torch.FloatTensor(weights)), requires_grad=True)
    biases = torch.nn.Parameter(use_cuda(torch.FloatTensor(biases)), requires_grad=True)

    convsp = spn.ConvSP(NCHANNELS, NKERNELS, NDIM, KERNEL_SIZE, DILATION, RADIUS)
    convsp.weight = weights
    convsp.bias = biases
    convsp = use_cuda(convsp)

    pred = undo_cuda(convsp(locs, data, density))
    np.testing.assert_array_almost_equal(pred.data.numpy(), ground_truth, decimal=3)

    def func(d, w, b):
        convsp.weight = w
        convsp.bias = b
        return (convsp(locs, d, density),)
    assert torch.autograd.gradcheck(func, (data, weights, biases), eps=1e-2, atol=1e-3)



if __name__ == '__main__':
    test_convsp()