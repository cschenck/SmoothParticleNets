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


def pyconvsp(qlocs, locs, data, weights, biases, kernel_fn, KERNEL_SIZE, RADIUS, DILATION, NKERNELS):
    w = spn.KERNEL_FN[kernel_fn]

    BATCH_SIZE = locs.shape[0]
    M = qlocs.shape[1]
    N = locs.shape[1]
    NDIM = locs.shape[-1]

    kernel_centers = (np.array(KERNEL_SIZE) - 1)/2
    ground_truth = np.zeros((BATCH_SIZE, M, NKERNELS), dtype=data.dtype)
    for b in range(BATCH_SIZE):
        for i in range(M):
            for j in range(N):
                dd = np.square(qlocs[b, i, :] - locs[b, j, :]).sum()
                nr = DILATION*max(KERNEL_SIZE)/2 + RADIUS
                if dd > nr*nr:
                    continue
                for k, idxs in enumerate(itertools.product(*[range(x) for x in KERNEL_SIZE[::-1]])):
                    dd = np.square(qlocs[b, i, :] + (idxs[::-1] - kernel_centers)*DILATION 
                        - locs[b, j, :]).sum()
                    if dd > RADIUS*RADIUS:
                        continue
                    ground_truth[b, i, :] += weights[:, :, k].dot(
                        w(np.sqrt(dd), RADIUS)*data[b, j, :])
    ground_truth += biases[np.newaxis, np.newaxis, :]
    return ground_truth


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
    N = 5
    M = 3
    NDIM = 2
    KERNEL_SIZE = (3, 1)
    RADIUS = 1.0
    DILATION = 0.05
    NCHANNELS = 2
    NKERNELS = 3

    np.random.seed(0)

    locs = np.random.rand(BATCH_SIZE, N, NDIM).astype(np.float32)
    qlocs = np.random.rand(BATCH_SIZE, M, NDIM).astype(np.float32)
    data = np.random.rand(BATCH_SIZE, N, NCHANNELS).astype(np.float32)
    weights = np.random.rand(NKERNELS, NCHANNELS, np.prod(KERNEL_SIZE)).astype(np.float32)
    biases = np.random.rand(NKERNELS).astype(np.float32)

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

    for use_qlocs in (True, False):

        locs_t = torch.autograd.Variable(use_cuda(torch.FloatTensor(locs)), requires_grad=True)
        if use_qlocs:
            qlocs_t = torch.autograd.Variable(use_cuda(torch.FloatTensor(qlocs)), requires_grad=True)
        else:
            qlocs_t = None
        data_t = torch.autograd.Variable(use_cuda(torch.FloatTensor(data)), requires_grad=True)
        weights_t = torch.nn.Parameter(torch.FloatTensor(weights), requires_grad=True)
        biases_t = torch.nn.Parameter(torch.FloatTensor(biases), requires_grad=True)

        coll = use_cuda(spn.ParticleCollision(NDIM, 
            RADIUS + DILATION*max((k - 1)/2 for k in KERNEL_SIZE)))
        locs_t, data_t, idxs_t, neighbors_t = coll(locs_t, data_t, (qlocs_t if use_qlocs else None))

        for kernel_fn in spn.KERNEL_NAMES:
            print("\tTesting kernel %s (%s query locations)..." % 
                (kernel_fn, "with" if use_qlocs else "without"))
            ground_truth = pyconvsp((qlocs if use_qlocs else locs), locs, data, weights, biases, 
                kernel_fn, KERNEL_SIZE, RADIUS, DILATION, NKERNELS)

            convsp = spn.ConvSP(NCHANNELS, NKERNELS, NDIM, KERNEL_SIZE, DILATION, RADIUS,
                kernel_fn=kernel_fn)
            convsp.weight = weights_t
            convsp.bias = biases_t
            convsp = use_cuda(convsp)

            pred_t = undo_cuda(convsp(locs_t, data_t, neighbors_t, qlocs_t))
            np.testing.assert_array_almost_equal(pred_t.data.numpy(), ground_truth, decimal=3)

            dt = torch.autograd.Variable(data_t.data, requires_grad=True)
            lt = torch.autograd.Variable(locs_t.data, requires_grad=True)
            if use_qlocs:
                qt = torch.autograd.Variable(qlocs_t.data, requires_grad=True)
            wt = torch.nn.Parameter(weights_t.data, requires_grad=True)
            bt = torch.nn.Parameter(biases_t.data, requires_grad=True)
            # Use pyconvsp to allow for double precision when computing numeric grads.
            def func_numerical(l, d, w, b, q=None):
                return (torch.autograd.Variable(torch.from_numpy(
                    pyconvsp((q.data.cpu().numpy() if use_qlocs else l.data.cpu().numpy()), 
                        l.data.cpu().numpy(), 
                        d.data.cpu().numpy(), w.data.cpu().numpy(), b.data.cpu().numpy(), 
                        kernel_fn, KERNEL_SIZE, RADIUS, DILATION, NKERNELS))),)
            def func_analytical(l, d, w, b, q=None):
                convsp.weight = w
                convsp.bias = b
                return (convsp(l, d, neighbors_t, (q if use_qlocs else None)),)
            assert gradcheck(func_analytical, 
                ((lt, dt, wt, bt, qt) if use_qlocs else (lt, dt, wt, bt,)), 
                eps=1e-4, atol=1e-3, rtol=1e-2, func_numerical=func_numerical, use_double=True)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', dest='cpu', action="store_true", default=True)
    parser.add_argument('--no-cpu', dest='cpu', action="store_false")
    parser.add_argument('--cuda', dest='cuda', action="store_true", default=True)
    parser.add_argument('--no-cuda', dest='cuda', action="store_false")
    args = parser.parse_args()
    test_convsp(cpu=args.cpu, cuda=args.cuda)