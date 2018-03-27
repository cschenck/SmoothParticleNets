import os
import sys
# Add path to python source to path.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))
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


def test_particlecollision(cpu=True, cuda=True):
    if cpu:
        print("Testing CPU implementation of ParticleCollision...")
        eval_particlecollision(cuda=False)
        print("CPU implementation passed!")
        print("")

    if cuda:
        if pytest_args.with_cuda:
            print("Testing CUDA implementation of ParticleCollision...")
            eval_particlecollision(cuda=True)
            print("CUDA implementation passed!")
        else:
            print("Not compiled with CUDA, skipping CUDA test.")

def eval_particlecollision(cuda=False):
    BATCH_SIZE = 2
    N = 100
    M = 77
    NDIM = 2
    RADIUS = 0.2
    NCHANNELS = 2

    np.random.seed(0)

    locs = np.random.rand(BATCH_SIZE, N, NDIM).astype(np.float32)
    qlocs = np.random.rand(BATCH_SIZE, M, NDIM).astype(np.float32)
    data = np.random.rand(BATCH_SIZE, N, NCHANNELS).astype(np.float32)

    gt_neighbors = np.ones((BATCH_SIZE, M, N), dtype=int)*-1
    for b in range(BATCH_SIZE):
        for i in range(M):
            for j in range(N):
                d = np.square(qlocs[b, i, :] - locs[b, j, :]).sum()
                if d <= RADIUS*RADIUS:
                    nc = min(np.where(gt_neighbors[b, i, :] < 0)[0])
                    gt_neighbors[b, i, nc] = j

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

    olocs = locs
    oqlocs = qlocs
    odata = data
    locs = torch.autograd.Variable(use_cuda(torch.FloatTensor(locs.copy())), 
        requires_grad=False)
    qlocs = torch.autograd.Variable(use_cuda(torch.FloatTensor(qlocs.copy())), 
        requires_grad=False)
    data = torch.autograd.Variable(use_cuda(torch.FloatTensor(data.copy())), 
        requires_grad=False)

    coll = spn.ParticleCollision(NDIM, RADIUS, max_collisions=N)
    convsp = use_cuda(coll)

    vlocs, vdata, vidxs, vneighbors = coll(locs, data, qlocs)
    
    idxs = undo_cuda(vidxs).data.numpy().astype(int)
    neighbors = undo_cuda(vneighbors).data.numpy().astype(int)
    nlocs = undo_cuda(vlocs).data.numpy()
    ndata = undo_cuda(vdata).data.numpy()

    # First make sure all the indexes are in idxs.
    for b in range(BATCH_SIZE):
        for i in range(N):
            assert i in idxs[b, :]

    # Next make sure locs and data are in the order idxs says they're in.
    for b in range(BATCH_SIZE):
        for i, j in enumerate(idxs[b, :]):
            assert all(olocs[b, j, :] == nlocs[b, i, :])
            assert all(odata[b, j, :] == ndata[b, i, :])

    # Check the neighbor list.
    for b in range(BATCH_SIZE):
        for i in range(M):
            for j in neighbors[b, i, :]:
                if j < 0:
                    break
                assert idxs[b, j] in gt_neighbors[b, i, :]
            for j in gt_neighbors[b, i, :]:
                if j < 0:
                    break
                jj = np.where(idxs[b, :] == j)[0][0]
                assert jj in neighbors[b, i, :]

    # Finally put the locations and data back in their original order.
    reorder = use_cuda(spn.ReorderData(reverse=True))
    vlocs, vdata = reorder(vidxs, vlocs, vdata)
    assert np.all(undo_cuda(vlocs).data.numpy() == olocs)
    assert np.all(undo_cuda(vdata).data.numpy() == odata)

    # Test gradients.
    def func(l, d, q):
        return coll(l, d, q)[:2]
    assert gradcheck(func, (locs, data, qlocs), eps=1e-2, atol=1e-3)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', dest='cpu', action="store_true", default=True)
    parser.add_argument('--no-cpu', dest='cpu', action="store_false")
    parser.add_argument('--cuda', dest='cuda', action="store_true", default=True)
    parser.add_argument('--no-cuda', dest='cuda', action="store_false")
    args = parser.parse_args()
    test_particlecollision(cpu=args.cpu, cuda=args.cuda)