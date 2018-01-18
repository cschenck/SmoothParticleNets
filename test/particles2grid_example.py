"""
This file is a complete example of training a network using the particles2grid and
grid2particles layers. The input data is a set of particles, each with some number
of associated values. The output is then a new set of values for each particle. The
network uses the particles2grid layer to convert the set of input values at each
particle to a grid of values. Then it applies standard 3D convolutions. Finally, it
uses the grid2particles layer to convert the resulting values from a grid back to
values for each particle.

The data used to train and test the network is randomly generated. Simply running
this file will begin the training process. Use ctrl+c to stop training. If you have
tensorflow, matplotlib, and opencv installed, then you can specify a directroy to
save a summary file that can then be viewed using tensorboard.
"""

import os
import sys
# Add path to python source to path.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import SmoothParticleNets as spn

import multiprocessing
import numpy as np
import time
import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as tud

try:
    from tblogger import TBLogger
except:
    import traceback
    traceback.print_exc()
    print("Unable to import the tensboard logger. Going forward without logging.")
    TBLogger = None

MASS = 0.1
RADIUS = 1.0

# The kernel function for SPH.
def w(x, h=1):
    x = np.sqrt((x**2).sum())
    return 1.0/np.pi*(0.25*max(0, h - x)**3 - max(0, h/2.0 - x)**3)/(h**3/(8*np.pi))

# The derivative of the function above.
def dw(x, h=1):
    d = np.sqrt((x**2).sum())
    ret = np.zeros(x.shape, dtype=x.dtype)
    if d < h:
        ret += (3*h*x/(2*np.pi) - 3*x*d/(4*np.pi) - 
            3*h*h/(4*np.pi)*(0 if d == 0 else x/d))
    if d < h/2.0:
        ret -= (3*h*x/np.pi - 3*x*d/np.pi - 
            3*h*h/(4*np.pi)*(0 if d == 0 else x/d))
    return ret/(h**3/(8*np.pi))



# Randomly generate a set of particles with xyz vectors associated with each. The input
# vairable N is the number of particles to generate. The function returns a tuple with
# the following elements:
#   -locs: a Nx4 array of the locations of each particle.
#   -data: a Nx1 array of the data value associated with each particle.
#   -density: a N length array of the density at each particle.
#   -vel: a Nx3 array of the xyz vector associated with each particle. This is computed
#         from the derivative of the SPH field induced by the data variable.
# Note tha the PROCESS_NAME global variables is due to a bug with multiprocessing where
# when forking the processes for generating data, they all get the same random seed,
# resulting in identical generated data. By using the name of the process, this can be
# avoided.
PROCESS_NAME = ""
def gen_data(N=10):
    global PROCESS_NAME
    if PROCESS_NAME != multiprocessing.current_process().name:
        PROCESS_NAME = multiprocessing.current_process().name
        np.random.seed(PROCESS_NAME.__hash__() % (2**32 - 1))
    truelocs = np.array([
                         [0, 0, 0],
                         [0, 0, 1],
                         [0, 1, 0],
                         [0, 1, 1],
                         [1, 0, 0],
                         [1, 0, 1],
                         [1, 1, 0],
                         [1, 1, 1],
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


# Class for the pytorch DataLoader.
class GeneratedDataset(tud.Dataset):
    def __init__(self, N=10, size=1000):
        self.N = N
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        locs, data, density, vel = gen_data(N=self.N)
        return {"locs" : locs, "data" : data, "density" : density, "vel" : vel, "key" : key}


# The network. The input is first passed to the particles2grid layer, then through NCONVS
# 3D convolution layers (each followed by a PReLU), then through the grid2particles layer.
class SimpleSmoothParticleNet(nn.Module):
    def __init__(self, grid_dims, grid_lower, grid_steps, radius, 
                 indata_dims, nKernels, outdata_dims):
        super(SimpleSmoothParticleNet, self).__init__()
        self.particles2grid = spn.Particles2Grid(grid_dims, grid_lower, grid_steps, radius)
        self.grid2particles = spn.Grid2Particles(grid_lower, grid_steps)
        NCONVS = 4
        self.convs = ([nn.Conv3d(indata_dims, nKernels, kernel_size=3, stride=1, padding=1),] +
            [nn.Conv3d(nKernels, nKernels, kernel_size=3, stride=1, padding=1) 
                for _ in range(NCONVS - 2)] +
            [nn.Conv3d(nKernels, outdata_dims, kernel_size=3, stride=1, padding=1)])
        self.nonlins = [nn.PReLU() for _ in range(NCONVS - 1)]
        for i in range(len(self.convs)):
            exec("self.conv%d = self.convs[%d]" % (i, i))
            if i < len(self.nonlins):
                exec("self.nonlin%d = self.nonlins[%d]" % (i, i))

    def forward(self, locs, data, density):
        grid = self.particles2grid(locs, data, density)
        grid = grid.permute(0, 4, 1, 2, 3)
        for i in range(len(self.convs)):
            grid = self.convs[i](grid)
            if i < len(self.nonlins):
                grid = self.nonlins[i](grid)
        grid = grid.permute(0, 2, 3, 4, 1)
        self.grid = grid
        self.locs = locs
        self.density = density
        output = self.grid2particles(grid, locs)
        return output

    # While it is possible to backpropagate through the grid2particles layer, a
    # more dense training signal can be acquired by instead taking the ground
    # truth over a set of particles and converting that to a grid, then taking
    # the loss with respect to the final grid output by the network prior to
    # converting it to a set of particles.
    def compute_loss(self, criterion, gt):
        return criterion(self.grid, self.particles2grid(self.locs, gt, self.density))


def generate_predictions(net, locs, data, density):
    pred = net(locs, data, density)
    return pred

def eval_criterion(net, criterion, pred, locs, density, vel):
    return criterion(pred, vel).cpu().data.numpy()[0]

def eval_set(net, criterion, locs, data, density, vel):
    pred = generate_predictions(net, locs, data, density)
    return eval_criterion(net, criterion, pred, locs, density, vel)

def numpybatch2torchbatch(batch, requires_grad=True):
    return [Variable(torch.FloatTensor(np.array(x)).cuda(), 
                requires_grad=requires_grad) for x in batch]


# Visualize the progress of training using tensorboard.
def viz(tblogger, iteration, loss, net, criterion, test_set):
    if tblogger is None or iteration % 30 != 0:
        return
    tblogger.scalar_summary("Training Loss", loss.cpu().data.numpy()[0], iteration)
    pred = generate_predictions(net, *test_set[:-1])
    tblogger.scalar_summary("Testing Loss", 
        eval_criterion(net, criterion, pred, test_set[0], test_set[2], test_set[3]), iteration)
    for idx in range(2):
        locs = test_set[0].data.cpu().numpy()[idx, ..., :3]
        pred_grid = net.grid
        if pred.data.size()[-1] > 1:
            tblogger.vecdiff_summary(str(idx) + "-Test Particles", iteration, locs,
                test_set[-1].data.cpu().numpy()[idx, ...], 
                pred.data.cpu().numpy()[idx, ...],
                axlimits=[(0, 1), (0, 1), (0, 1)], scale=0.5)
            tblogger.scatter3d_summary(str(idx) + "-Test Particles Input", iteration, locs, 
                data=test_set[1].data.cpu().numpy()[idx, ...], 
                axlimits=[(0, 1), (0, 1), (0, 1)], titles=["Input",])
        else:
            data = np.concatenate((test_set[1].data.cpu().numpy()[idx, ...], 
                test_set[-1].data.cpu().numpy()[idx, ...], 
                pred.data.cpu().numpy()[idx, ...]), axis=-1)
            tblogger.scatter3d_summary(str(idx) + "-Test Particles", iteration, locs, data=data, 
                axlimits=[(0, 1), (0, 1), (0, 1)], titles=["Input", "Ground Truth", "Predictions"])
        
        pred_grid = pred_grid.data.cpu().numpy()[idx, ...]
        data_grid = net.particles2grid(test_set[0], test_set[1], 
                                        test_set[2]).data.cpu().numpy()[idx,...]
        vel_grid = net.particles2grid(test_set[0], test_set[-1], 
                                        test_set[2]).data.cpu().numpy()[idx,...]
        if pred.data.size()[-1] > 1:
            if data_grid.shape[-1] < pred_grid.shape[-1]:
                data_grid = np.concatenate((data_grid, np.zeros(data_grid.shape[:-1] + 
                    (pred_grid.shape[-1] - data_grid.shape[-1],), dtype=data_grid.dtype)), axis=-1)
            titles = []
            for i, s in enumerate(["Input", "Ground Truth", "Predictions"]):
                titles += [s + "-" + str(j) for j in range(pred_grid.shape[-1])]
            shape = (3, pred_grid.shape[-1])
            #Normalize the outputs
            pred_grid *= 5
            vel_grid *= 5
            color_norms = (np.array([0]*3 + [-1]*6), np.array([1]*9))
        else:
            titles = ["Input", "Ground Truth", "Predictions"]
            shape = None
            color_norms = (0, 1)
        grid = np.concatenate([data_grid, vel_grid, pred_grid], axis=-1)
        tblogger.grid3d_summary(str(idx) + "-Test Grids", iteration, grid, [0.0 for _ in range(3)], 
            [1.0/pred_grid.shape[i] for i in range(3)], 
            titles=titles, transparency_scale=(0.25 - 0.2*RADIUS), shape=shape,
            color_norms=color_norms)

    tblogger.flush()


def main():
    gendataset = GeneratedDataset()
    dataloader = tud.DataLoader(gendataset, batch_size=32, shuffle=True, num_workers=8)
    grid_dims = np.array((50, 50, 50))
    
    test_set = numpybatch2torchbatch(zip(*[gen_data() for _ in range(32)]), requires_grad=False)
    net = SimpleSmoothParticleNet(grid_dims, (0, 0, 0), 1.0/grid_dims, RADIUS, 
        test_set[1].data.size()[-1], 32, test_set[-1].data.size()[-1]).cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-2)
    if TBLogger is not None:
        tblogger = TBLogger(sys.argv[1])
    else:
        tblogger = None

    print("Initial error: %15.7f" % eval_set(net, criterion, *test_set))

    epoc = 0
    iteration = 0
    while True:
        start = time.time()
        for i, d in enumerate(dataloader):
            locs = Variable(d['locs'].cuda(), requires_grad=False)
            data = Variable(d['data'].cuda(), requires_grad=False)
            density = Variable(d['density'].cuda(), requires_grad=False)
            vel = Variable(d['vel'].cuda(), requires_grad=False)
            pred = net(locs, data, density)
            loss = net.compute_loss(criterion, vel)
            print('\tITER[%2d] loss: %f' % (i, loss.cpu().data.numpy()[0]))
            optimizer.zero_grad()
            loss.backward()
            viz(tblogger, iteration, loss, net, criterion, test_set)
            optimizer.step()
            iteration += 1
        t = time.time() - start
        print("EPOC %d: %15.7f (%10.3fs)" % (epoc, eval_set(net, criterion, *test_set), t))
        epoc += 1




if __name__ == '__main__':
    main()