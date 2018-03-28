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
try:
    import pytest_args
except ImportError:
    print("Make sure to compile SmoothParticleNets before running tests.")
    raise

# The kernel function for SPH.
def w(x, h=1):
    return 1.0/np.pi*(0.25*max(0, h - x)**3 - max(0, h/2.0 - x)**3)/(h**3/(8*np.pi))


def test_convsp(cpu=True, cuda=True):
    if cpu:
        print("Testing CPU implementation of ConvSP...")
        eval_convsp(cuda=False)
        print("CPU implementation passed!")
        print("")

    if cuda:
        if pytest_args.with_cuda:
            print("Testing CUDA implementation of ConvSP...")
            eval_convsp(cuda=True)
            print("CUDA implementation passed!")
        else:
            print("Not compiled with CUDA, skipping CUDA test.")

def eval_convsp(cuda=False):
    BATCH_SIZE = 2
    N = 10
    M = 13
    NDIM = 2
    KERNEL_SIZE = (3, 5)
    RADIUS = 1.0
    DILATION = 0.05
    NCHANNELS = 2
    NKERNELS = 3
    MASS = 1.0

    np.random.seed(0)

    locs = np.random.rand(BATCH_SIZE, N, NDIM)
    qlocs = np.random.rand(BATCH_SIZE, M, NDIM)
    data = np.random.rand(BATCH_SIZE, N, NCHANNELS)
    weights = np.random.rand(NKERNELS, NCHANNELS, np.prod(KERNEL_SIZE))
    biases = np.random.rand(NKERNELS)

    kernel_centers = (np.array(KERNEL_SIZE) - 1)/2
    ground_truth = np.zeros((BATCH_SIZE, M, NKERNELS), dtype=np.float32)
    for b in range(BATCH_SIZE):
        for i in range(M):
            for j in range(N):
                d = np.square(qlocs[b, i, :] - locs[b, j, :]).sum()
                nr = DILATION*max(KERNEL_SIZE)/2 + RADIUS
                if d > nr*nr:
                    continue
                for k, idxs in enumerate(itertools.product(*[range(x) for x in KERNEL_SIZE[::-1]])):
                    d = np.square(qlocs[b, i, :] + (idxs[::-1] - kernel_centers)*DILATION 
                        - locs[b, j, :]).sum()
                    if d > RADIUS*RADIUS:
                        continue
                    ground_truth[b, i, :] += weights[:, :, k].dot(
                        w(np.sqrt(d), h=RADIUS)*data[b, j, :])
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

    locs = torch.autograd.Variable(use_cuda(torch.FloatTensor(locs)), requires_grad=False)
    qlocs = torch.autograd.Variable(use_cuda(torch.FloatTensor(qlocs)), requires_grad=False)
    data = torch.autograd.Variable(use_cuda(torch.FloatTensor(data)), requires_grad=True)
    weights = torch.nn.Parameter(torch.FloatTensor(weights), requires_grad=True)
    biases = torch.nn.Parameter(torch.FloatTensor(biases), requires_grad=True)

    coll = use_cuda(spn.ParticleCollision(NDIM, 
        RADIUS + DILATION*max((k - 1)/2 for k in KERNEL_SIZE)))
    locs, data, idxs, neighbors = coll(locs, data, qlocs)
    # reorder = use_cuda(spn.ReorderData(reverse=False))
    # ground_truth = torch.autograd.Variable(use_cuda(torch.FloatTensor(ground_truth)), 
    #     requires_grad=False)
    # reorder(idxs, ground_truth)
    # ground_truth = undo_cuda(ground_truth).data.numpy()

    convsp = spn.ConvSP(NCHANNELS, NKERNELS, NDIM, KERNEL_SIZE, DILATION, RADIUS)
    convsp.weight = weights
    convsp.bias = biases
    convsp = use_cuda(convsp)
    if cuda:
        # Set convsp's amount of shared memory low enough so that the convsp cuda
        # implementation has to split the particles into blocks.
        device_id = torch.cuda.current_device()
        nshared_device_mem = spn._ext.spnc_get_shared_mem_size(device_id)
        f32 = np.dtype('float32').itemsize
        fixedmem = (NKERNELS*NCHANNELS*np.prod(KERNEL_SIZE)*f32)
        fixedmem += NDIM*f32
        fixedmem += NDIM*f32
        memperparticle = NDIM*f32
        memperparticle += NCHANNELS*f32
        memperparticle += f32
        memperparticle += NKERNELS*f32
        block_size = N//3
        convsp.device_id = device_id
        convsp.nshared_device_mem = min(nshared_device_mem, 
            fixedmem + block_size*memperparticle*2)


    pred = undo_cuda(convsp(locs, data, neighbors, qlocs))
    np.testing.assert_array_almost_equal(pred.data.numpy(), ground_truth, decimal=3)

    if cuda:
        # Add more device memory for doing backwards passes.
        fixedmem += (NKERNELS*NCHANNELS*np.prod(KERNEL_SIZE)*f32)
        memperparticle += NCHANNELS*f32
        block_size = N//3
        convsp.nshared_device_mem = min(nshared_device_mem, 
            fixedmem + block_size*memperparticle*2)


    neighbors = torch.autograd.Variable(neighbors.data, requires_grad=False)
    data = torch.autograd.Variable(data.data, requires_grad=True)
    locs = torch.autograd.Variable(locs.data, requires_grad=False)
    def func(l, d, w, b, q):
        convsp.weight = w
        convsp.bias = b
        return (convsp(l, d, neighbors, q),)
    assert gradcheck(func, (locs, data, weights, biases, qlocs), eps=1e-2, atol=1e-3)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', dest='cpu', action="store_true", default=True)
    parser.add_argument('--no-cpu', dest='cpu', action="store_false")
    parser.add_argument('--cuda', dest='cuda', action="store_true", default=True)
    parser.add_argument('--no-cuda', dest='cuda', action="store_false")
    args = parser.parse_args()
    test_convsp(cpu=args.cpu, cuda=args.cuda)