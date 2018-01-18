
"""
This test is designed to show the non-symmetry of the particles2grid layer. Essentially it uses
a nearest-neighbor method for computing the value at each grid cell based on a set of
particles (refer to the documentation for that layer for more details). This is inherently non-
symmetric, and so any set of particles fed through it and then inverted back to particles
from the grid will be different than the original particles.

When run, 2 figures will appear. The green lines show a randomly generated set of particles
each with an associated xyz vector. The blue lines show the results after converting the
particles to a grid then back to particles again. Ideally these would be identical, but
here they are not. The red line shows the disparity between the two. This file requires
matplotlib to run.
"""

import os
import sys
# Add path to python source to path.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import SmoothParticleNets as spn

try:
	import matplotlib.pyplot as plt
	import matplotlib.colors
	from mpl_toolkits.mplot3d import Axes3D
except:
	import traceback
	traceback.print_exc()
	print("This test requires matplotlib. Please install and run again.")
	sys.exit(1)
import numpy as np

# Stop code execution and open a python terminal at the place this is called.
# Credit to effbot.org/librarybook/code.htm for loading variables into current namespace
def keyboard(banner=None):
    import code, sys

    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back

    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)

    code.interact(banner=banner, local=namespace)

# Python version of the particles2grid layer. Arguments are the same. Very slow.
def py_particles2grid(locs, data, density, grid_size, grid_lower, grid_steps, radius):
    if len(locs.shape) > 2:
        ret = np.zeros((locs.shape[0],) + grid_size + data.shape[3:], dtype=data.dtype)
        has_batch = True
    else:
        ret = np.zeros((1,) + grid_size + data.shape[3:], dtype=data.dtype)
        has_batch = False
        locs = np.expand_dims(locs, 0)
        data = np.expand_dims(data, 0)
        density = np.expand_dims(density, 0)
    for b in range(ret.shape[0]):
        for ii in range(ret.shape[1]):
            for jj in range(ret.shape[2]):
                for kk in range(ret.shape[3]):
                    idx = (b, ii, jj, kk)
                    r1 = np.array([(idx[i+1] + 0.5)*grid_steps[i] + grid_lower[i] 
                        for i in range(3)])
                    for i, r2 in enumerate(locs[b, ...]):
                        ret[idx] += (1.0/(r2[-1]*density[b, i])*data[b, i, ...]*
                                        w(r2[:3] - r1, h=radius))
    if not has_batch:
        ret = ret[0, ...]
    return ret

# Python version of the grid2particles layer that uses an RGB kernel instead of
# trilinear interpolation. Arguments are the same. Very slow.
def py_grid2particles(grid, locs, grid_lower, grid_steps, radius):
    if len(locs.shape) > 2:
        if len(grid.shape) == 4:
            grid = np.expand_dims(grid, -1)
        has_batch = True
    else:
        if len(grid.shape) == 3:
            grid = np.expand_dims(grid, -1)
        has_batch = False
        locs = np.expand_dims(locs, 0)
        grid = np.expand_dims(grid, 0)
    grid_size = grid.shape[1:-1]
    batch_size = grid.shape[0]
    N = locs.shape[1]
    data_dims = grid.shape[-1]
    ret = np.zeros((batch_size, N, data_dims,), dtype=grid.dtype)
    weights = np.zeros((batch_size,N, 1,), dtype=np.float32)
    for b in range(batch_size):
        for i, r1 in enumerate(locs[b, :, :3]):
            for ii in range(grid_size[0]):
                for jj in range(grid_size[1]):
                    for kk in range(grid_size[2]):
                        r2 = np.array([(ii + 0.5)*grid_steps[0] + grid_lower[0],
                                       (jj + 0.5)*grid_steps[1] + grid_lower[1],
                                       (kk + 0.5)*grid_steps[2] + grid_lower[2]])
                        d = ((r1 - r2)**2).sum()
                        if d <= radius*radius:
                            ww = np.exp(-1.0*d/(2.0*(radius/2.0)**2))
                            ret[b, i, ...] += grid[b, ii, jj, kk, ...]*ww
                            weights[b, i, ...] += ww
    ret /= weights
    if not has_batch:
        ret = ret[0, ...]
    return ret

# Randomly generate a set of particles with xyz vectors associated with each. The input
# vairable N is the number of particles to generate. The function returns a tuple with
# the following elements:
#	-locs: a Nx4 array of the locations of each particle.
#	-data: a Nx1 array of the data value associated with each particle.
#	-density: a N length array of the density at each particle.
#	-vel: a Nx3 array of the xyz vector associated with each particle. This is computed
#		  from the derivative of the SPH field induced by the data variable.
def gen_data(N=10):
    truelocs = np.array([
                         [0.5, 0.5, 0.5],
                        ])
    truedata = np.round(np.random.rand(truelocs.shape[0], 1))

    locs = np.random.rand(N, 4).astype(np.float32)
    locs[:, -1] = 1.0/MASS
    # Density for true data
    density = np.zeros((truelocs.shape[0],), dtype=np.float32)
    for i, r1 in enumerate(truelocs):
        for r2 in truelocs:
            density[i] += MASS*w(r1 - r2, h=RADIUS)
    data = np.zeros((N, 1), dtype=np.float32)
    for i, r1 in enumerate(locs[:, :-1]):
        for j, r2 in enumerate(truelocs):
            data[i, ...] += MASS/density[j]*truedata[j, ...]*w(r1 - r2, h=RADIUS)

    # Density for data, not true data
    density = np.zeros((N,), dtype=np.float32)
    for i, r1 in enumerate(locs[:, :-1]):
        for r2 in locs[:, :-1]:
            density[i] += MASS*w(r1 - r2, h=RADIUS)
    vel = np.zeros(data.shape[:-1] + (3,), dtype=data.dtype)
    for i, r1 in enumerate(locs[:, :-1]):
        for j, r2 in enumerate(locs[:, :-1]):
            vel[i, ...] += MASS/density[j]*dw(r1 - r2, h=RADIUS)*data[j, ...]
    vel /= 2.0/RADIUS
    return locs, data, density, vel

# Generate a figure showing the difference between two sets of vectors for the same
# set of points. locs should be an Nx3 array of particle locations. _vec1 and _vec2
# should both be Nx3 arrays of xyz vectors. For each particle, the vector from 
# _vec1 will be drawn in green and the vector from _vec2 will be drawn in blue. The
# disparity between the two will be drawn in red.
def plot(locs, _vec1, _vec2, scale=0.5, axlimits=[(0, 1), (0, 1), (0, 1)]):
    _vec1 = _vec1.copy()
    _vec2 = _vec2.copy()
    fig = plt.figure()
    fig.set_size_inches(14, 10, forward=True)
    ax = fig.add_subplot(111, projection='3d')
    N = locs.shape[0]
    _vec1 *= scale
    _vec2 *= scale
    _vec3 = _vec1 - _vec2
    locs = np.concatenate((locs, locs, locs + _vec2), axis=0)
    colors = np.zeros((3*N, 4), dtype=np.float32)
    colors[:, 3] = 1.0
    colors[:N, 1] = 1.0
    colors[N:(2*N), 2] = 1.0
    colors[(2*N):, 0] = 1.0
    _vec = np.concatenate((_vec1, _vec2, _vec3), axis=0)
    ax.quiver(locs[:, 0], locs[:, 1], locs[:, 2], _vec[:, 0], 
        _vec[:, 1], _vec[:, 2], colors=colors, arrow_length_ratio=0.0,
        length=1.0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    if axlimits is not None:
        ax.set_xlim(*axlimits[0])
        ax.set_ylim(*axlimits[1])
        ax.set_zlim(*axlimits[2])
    fig.canvas.draw()
    fig.show()


def test():
    
    plt.ion()
    particles2grid = spn.Particles2Grid((50, 50, 50), (0, 0, 0), (1.0/50, 1.0/50, 1.0/50), RADIUS)
    grid2particles = spn.Grid2Particles((0, 0, 0), (1.0/50, 1.0/50, 1.0/50))
    test_set = numpybatch2torchbatch(zip(*[gen_data() for _ in range(32)]), requires_grad=False)
    locs = test_set[0].data.cpu().numpy()[0, ..., :3]
    vec1 = test_set[-1].data.cpu().numpy()[0, ...]
    vec1_grid = particles2grid(test_set[0], test_set[-1], test_set[2])
    vec2 = grid2particles(vec1_grid, test_set[0]).data.cpu().numpy()[0, ...]
    vec2py = py_grid2particles(vec1_grid.data.cpu().numpy()[0,...], test_set[0].data.cpu().numpy()[0,...], (0,0,0), (1.0/50, 1.0/50, 1.0/50), 0.05)
    plot(locs, vec1, vec2)
    plot(locs, vec1, vec2py)
    keyboard()


if __name__ == '__main__':
    test()