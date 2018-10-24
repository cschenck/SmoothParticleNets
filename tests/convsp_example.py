"""
This file is a complete example on how to use the ConvSP layer. It randomly generates a
dataset on the fly and uses it to train a network made entirely of ConvSP layers and
non-linearities. The data is a set of randomly generated particles in the unit cube, each
with a random 1-D feature. The ground truth that the network must match is the derivative
of the feature field induced by the particles wrt space. To train the network, simply run
this file.

If you have matplotlib, tensorflow (for tensorboard), and OpenCV installed, you can view
the log graphically in a web browser. Pass as the first argument a directory to store
the log file. The logger will log to a file in this directory. Then run tensorboard and
give it that directory to read from.
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
RADIUS = 0.1

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
			data[i, :] += MASS/density[j]*truedata[j, :]*w(r1 - r2, h=RADIUS)

	# Density for data, not true data
	density = np.zeros((N,), dtype=np.float32)
	for i, r1 in enumerate(locs[:, :-1]):
		for r2 in locs[:, :-1]:
			density[i] += MASS*w(r1 - r2, h=RADIUS)
	# vel = data
	vel = np.zeros((data.shape[0], 3), dtype=data.dtype)
	for i, r1 in enumerate(locs[:, :-1]):
	    for j, r2 in enumerate(locs[:, :-1]):
	        vel[i, :] += MASS/density[j]*dw(r1 - r2, h=RADIUS)*data[j, :]
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


class SimpleSmoothParticleNet(nn.Module):
	def __init__(self, in_channels, out_channels, radius, ndim, dilation):
		super(SimpleSmoothParticleNet, self).__init__()
		NCONVS = 1
		NKERNELS = 1
		self.convs = [spn.ConvSP((in_channels if i == 0 else NKERNELS), 
								  (NKERNELS if i < NCONVS - 1 else out_channels), 
								  ndim=ndim, kernel_size=3, dilation=dilation, 
								  radius=radius)
					  for i in range(NCONVS)]
		self.nonlins = [nn.PReLU() for _ in range(NCONVS - 1)]
		for i in range(len(self.convs)):
			exec("self.conv%d = self.convs[%d]" % (i, i))
			if i < len(self.nonlins):
				exec("self.nonlin%d = self.nonlins[%d]" % (i, i))

	def forward(self, locs, data, density):
		for i in range(len(self.convs)):
			data = self.convs[i](locs, data, density)
			if i < len(self.nonlins):
				data = self.nonlins[i](data)
		return data


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
		if pred.data.size()[1] > 1:
			gt = test_set[-1].data.cpu().numpy()[idx, ...]
			p = pred.data.cpu().numpy()[idx, ...]
			tblogger.vecdiff_summary(str(idx) + "-Test Particles", iteration, locs,
				gt, p, axlimits=[(0, 1), (0, 1), (0, 1)], scale=0.5)
			tblogger.scatter3d_summary(str(idx) + "-Test Particles Input", iteration, locs, 
				data=test_set[1].data.cpu().numpy()[idx, ...], 
				axlimits=[(0, 1), (0, 1), (0, 1)], titles=["Input",])
		else:
			data = np.concatenate((test_set[1].data.cpu().numpy()[idx, ...], 
				test_set[-1].data.cpu().numpy()[idx, ...], 
				pred.data.cpu().numpy()[idx, ...]), axis=0)
			tblogger.scatter3d_summary(str(idx) + "-Test Particles", iteration, locs, data=data, 
				axlimits=[(0, 1), (0, 1), (0, 1)], titles=["Input", "Ground Truth", "Predictions"])

	tblogger.flush()


def main():
	gendataset = GeneratedDataset()
	dataloader = tud.DataLoader(gendataset, batch_size=32, shuffle=True, num_workers=8)
	grid_dims = np.array((50, 50, 50))
	
	test_set = numpybatch2torchbatch(zip(*[gen_data() for _ in range(32)]), requires_grad=False)
	net = SimpleSmoothParticleNet(test_set[1].data.size()[2], test_set[-1].data.size()[2],
		RADIUS, test_set[0].data.size()[-1]-1, 0.05).cuda()
	[p.data.normal_(0, 1) for p in net.parameters()]

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
			loss = criterion(pred, vel)
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