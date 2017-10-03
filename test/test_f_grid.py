

import os
import sys
# Add path to python source to path.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import SmoothParticleNets as spn
import numpy as np
import torch
import torch.autograd



def test_FGrid():
    value_grid = torch.FloatTensor([[0, 1, 2],
                                    [3, 4, 5],
                                    [6, 7, 8]])
    locs = torch.FloatTensor([[0, 1],
                              [1, 2],
                              [2, 0],
                              [0, 2],
                              [0, 2],
                              [-1, 0]])
    output = torch.FloatTensor([1,
                                5,
                                6,
                                2,
                                2,
                                0])
    grad_output = torch.FloatTensor([1,
                                     2,
                                     3,
                                     4,
                                     5,
                                     6])
    grad_input = torch.FloatTensor([[0, 1, 9],
                                    [0, 0, 2],
                                    [3, 0, 0]])
    value_grid = torch.autograd.Variable(value_grid.cuda(), requires_grad=True)
    locs = torch.autograd.Variable(locs.cuda(), requires_grad=False)
    output = torch.autograd.Variable(output, requires_grad=False)
    grad_output = torch.autograd.Variable(grad_output, requires_grad=False)
    grad_input = torch.autograd.Variable(grad_input, requires_grad=False)

    fgrid = spn.FGrid()

    _output = fgrid(locs, value_grid).cpu()
    np.testing.assert_array_equal(_output.data.numpy(), output.data.numpy())

    # Make the derivatices be grad_output, assuming the forward pass was correct.
    loss = torch.sum(0.5*(-grad_output + output - _output)**2)
    loss.backward()
    np.testing.assert_array_equal(value_grid.grad.data.cpu().numpy(), grad_input.data.numpy())
