

import os
import sys
# Add path to python source to path.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import SmoothParticleNets as spn
import numpy as np
import torch
import torch.autograd



def test_particles2grid():
    grid_lower = (1, 1, 1)
    grid_dims = (3, 3, 3)
    grid_steps = (2, 2, 2)
    radius = 1.8*2
    locs = np.array([[3, 1, 3, 1.0/10.0],
                     [3, 5, 4, 1.0/10.0],
                     [6, 2, 1, 1.0/10.0],
                     [6, 6, 6, 1.0/10.0],
                    ])
    data = np.array([[1.0, 100.0],
                     [2.0, 700.0],
                     [5.0, 50.0],
                     [4.5, 679.0],
                    ])
    density = np.zeros(locs.shape[0], dtype=locs.dtype)
    for i in range(locs.shape[0]):
        for j in range(locs.shape[0]):
            d = sum([np.square(locs[i, k] - locs[j, k]) for k in range(3)])
            if d <= np.square(radius):
                d = np.sqrt(d)/radius*2
                density[i] += locs[j, -1]*(np.power(2 - d, 3)/4.0 - np.power(1 - d, 3)*(1 if d < 1 else 0))/np.pi
    grid = np.zeros(grid_dims + (data.shape[-1],), dtype=data.dtype)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                for p in range(locs.shape[0]):
                    x, y, z = [([i, j, k][ii] + 0.5)*grid_steps[ii] + grid_lower[ii] for ii in range(3)]
                    d = sum([np.square(locs[p, ii] - [x, y, z][ii]) for ii in range(3)])
                    if d <= np.square(radius):
                        d = np.sqrt(d)/radius*2
                        for dd in range(data.shape[-1]):
                            w = (locs[p, -1]/density[p]*
                                (np.power(2 - d, 3)/4.0 - np.power(1 - d, 3)*(1 if d < 1 else 0))/np.pi)
                            # print("python: [%d] (%d, %d, %d) %10f %10f" % (p, i, j, k, w, data[p, dd]))
                            grid[i, j, k, dd] += (w*data[p, dd])


    locs = torch.autograd.Variable(torch.FloatTensor(locs).cuda())
    data = torch.autograd.Variable(torch.FloatTensor(data).cuda())
    density = torch.autograd.Variable(torch.FloatTensor(density).cuda())
    grid = torch.autograd.Variable(torch.FloatTensor(grid))

    particles2grid = spn.Particles2Grid(grid_dims, grid_lower, grid_steps, radius)

    _output = particles2grid(locs, data, density).cpu()
    # for i in range(grid.data.numpy().shape[0]):
    #     for j in range(grid.data.numpy().shape[1]):
    #         for k in range(grid.data.numpy().shape[2]):
    #             s = "["
    #             for ii in range(grid.data.numpy().shape[3]):
    #                 s += "%10f," % grid.data.numpy()[i, j, k, ii]
    #             s += "], ["
    #             for ii in range(_output.data.numpy().shape[3]):
    #                 s += "%10f," % _output.data.numpy()[i, j, k, ii]
    #             s += "]"
    #             print(s)
    np.testing.assert_array_almost_equal(_output.data.numpy(), grid.data.numpy(), decimal=3)

if __name__ == '__main__':
    test_particles2grid()
