

import itertools
import os
import sys
# Add path to python source to path.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import SmoothParticleNets as spn
import math
import numpy as np
import Queue
import torch
import torch.autograd

# Get the RegularGridInterpolator from the other test file.
from test_grid2particles import RegularGridInterpolator

def construct_sdf(sdf_shape, nugget, width):
    scale = [1.0*width[i]/sdf_shape[i] for i in range(len(width))]
    ret = np.ones(sdf_shape, dtype=np.float32)*np.finfo(np.float32).max
    ret[nugget] = 0
    return prop_sdf(ret, [(0, nugget)], scale=scale) - np.min(sdf_shape)/3.0

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


def test_sdfs2grid():
    grid_lower = (1, 1, 1)
    grid_dims = (3, 3, 3)
    grid_steps = (2, 2, 2)
    
    sdf_widths = np.array([[4, 4, 4],
                           [6, 4, 8],
                           [5, 5, 5],
                          ], dtype=np.float32)
    sdf1 = construct_sdf((5, 5, 5), (1, 2, 3), sdf_widths[0, :])
    sdf2 = construct_sdf((6, 4, 8), (1, 0, 2), sdf_widths[1, :])
    sdf3 = construct_sdf((20, 20, 20), (5, 15, 5), sdf_widths[2, :])
    
    l = max(np.prod(x.shape) for x in [sdf1, sdf2, sdf3])
    sdfs = np.array([np.concatenate((x.flatten(), np.zeros((l - np.prod(x.shape),))))
        for x in [sdf1, sdf2, sdf3]], dtype=np.float32)
    sdf_poses = np.array([[1, 1, 1] + eulerToQuaternion([0, 0, math.pi/8]),
                          [2, 2, 2] + eulerToQuaternion([0, 0, 0]),
                          [2, 1, 3] + eulerToQuaternion([math.pi/2, 0, math.pi/4]),
                         ], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=int)
    sdf_shapes = np.array([x.shape for x in [sdf1, sdf2, sdf3]])
    

    sdf_fns = [RegularGridInterpolator([np.linspace(0.5*ww/y, ww - 0.5*ww/y, y) for y, ww in zip(x.shape, w)], x, 
                                        bounds_error=False, fill_value=np.finfo(np.float32).max) 
                for x, w in zip([sdf1, sdf2, sdf3], sdf_widths)]
    grid = np.ones(grid_dims, dtype=np.float32)*np.finfo(np.float32).max
    grid_idxs = np.array(list(itertools.product(*[range(x) for x in grid_dims])))
    grid_coords = grid_idxs*grid_steps + grid_lower + 0.5*np.array(grid_steps)
    sdf_coords = [np.array([pointByQuaternion(pt - p[:3], quaternionConjugate(p[3:])) 
                             for pt in grid_coords]) 
                    for p in sdf_poses]
    sdf_values = np.array([sdf_fns[i](sdf_coords[i]) for i in range(len(sdf_coords))]).min(axis=0)
    for i, idx in enumerate(grid_idxs):
        grid[tuple(idx)] = sdf_values[i]
    
    filled_idxs = grid_idxs[np.where(sdf_values < np.finfo(np.float32).max), :][0, ...]
    filled_values = sdf_values[np.where(sdf_values < np.finfo(np.float32).max)]
    import cutil
    cutil.keyboard()
    


#function taken from:
#http://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion
def quaternionMult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return [x, y, z, w]
   
#function taken from:
#http://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion
def quaternionConjugate(q):
    x, y, z, w = q
    return [-x, -y, -z, w]
    
def pointByQuaternion(point, quaternion):
    if len(point) == 3:
        x, y, z = point
        point = [x, y, z, 0.0]
    return quaternionMult(quaternionMult(quaternion, point), quaternionConjugate(quaternion))[:3]

def eulerToQuaternion(euler):
    #heading, attitude, bank = euler
    #heading, bank, attitude = euler
    #attitude, heading, bank = euler
    #attitude, bank, heading = euler
    bank, heading, attitude = euler
    c1 = math.cos(heading/2.0)
    s1 = math.sin(heading/2.0)
    c2 = math.cos(attitude/2.0)
    s2 = math.sin(attitude/2.0)
    c3 = math.cos(bank/2.0)
    s3 = math.sin(bank/2.0)
    c1c2 = c1*c2
    s1s2 = s1*s2
    w = c1c2*c3 - s1s2*s3
    x = c1c2*s3 + s1s2*c3
    y = s1*c2*c3 + c1*s2*s3
    z = c1*s2*c3 - s1*c2*s3
    m = math.sqrt(x**2 + y**2 + z**2 + w**2)
    x /= m
    y /= m
    z /= m
    w /= m
    return [x, y, z, w]
    

if __name__ == '__main__':
    test_sdfs2grid()
