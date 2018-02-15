

import os
import sys
# Add path to python source to path.
sys.path.append(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import SmoothParticleNets as spn
import argparse
import itertools
import numpy as np
import pickle
import time
import torch
from torch.autograd import Variable

KERNEL_SIZES = [
	(1,1,1),
	(3,1,1),
	(3,3,3),
]

CHANNEL_SIZES = range(1, 16)
RADIUS = 0.025
DILATION = 0.0125
MAX_GRID_SIZE = 64

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', required=True, action="store", type=str)
parser.add_argument('--cuda', action="store_true", default=False)
args = parser.parse_args()

def cudafy(x):
	if args.cuda:
		return x.cuda()
	else:
		return x

print("Loading dataset...")
dataset = pickle.load(open(args.datapath, "rb"))

print("Computing hash grid for dataset...")
nr = RADIUS + DILATION*((max([max(kernel_size) for kernel_size in KERNEL_SIZES]) - 1)//2)
for i in range(len(dataset)):
	sys.stdout.write("dataset[%d]: " % i)
	sys.stdout.flush()
	dataset[i]['cellIdxs'] = torch.from_numpy(np.zeros(dataset[i]['locs'].size()[:-1], 
												dtype=np.float32))
	dataset[i]['originalIndex'] = torch.from_numpy(np.zeros(dataset[i]['locs'].size()[:-1], 
													dtype=np.float32))
	dataset[i]['cellStart'] = torch.from_numpy(np.zeros((dataset[i]['locs'].size()[0], 
		MAX_GRID_SIZE**(dataset[i]['locs'].size()[-1] - 1)), dtype=np.float32))
	dataset[i]['cellEnd'] = torch.from_numpy(np.zeros((dataset[i]['locs'].size()[0], 
		MAX_GRID_SIZE**(dataset[i]['locs'].size()[-1] - 1)), dtype=np.float32))
	dataset[i]['gridShape'] = torch.from_numpy(np.zeros((dataset[i]['locs'].size()[0], 
		dataset[i]['locs'].size()[-1] - 1), dtype=np.float32))
	for b in range(dataset[i]['locs'].size()[0]):
		sys.stdout.write("%d,"%b)
		sys.stdout.flush()
		locs = dataset[i]['locs'][b, ..., :-1]
		N = locs.size()[0]
		minl = locs.min(dim=0)[0]
		maxl = locs.max(dim=0)[0]
		gridShape = torch.clamp(torch.ceil((maxl - minl)/nr), min=0, max=MAX_GRID_SIZE)
		gridCumShape = torch.from_numpy(np.concatenate(
			(np.cumprod(gridShape.numpy()[::-1])[1::-1], [1,])).astype(np.float32))
		dataset[i]['cellIdxs'][b, ...] = (torch.floor((locs - minl)/(maxl - minl)*gridShape)
											*gridCumShape).sum(dim=-1)
		dataset[i]['originalIndex'][b, ...] = torch.from_numpy(np.array(range(locs.shape[0]), 
																dtype=np.float32))
		to_reorder = ['cellIdxs', 'originalIndex', 'locs', 'data', 'density']
		def make2d(a):
			if len(a.size()) < 2:
				ret = a.view(a.size()[0], 1)
			else:
				ret = a
			if ret.size()[0] != N:
				ret = ret.transpose(0,1)
			return ret
		all_data = torch.cat([make2d(dataset[i][k][b, ...]) for k in to_reorder], dim=1)
		all_data = all_data[torch.sort(all_data[:, 0])[1], :]
		dataset[i]['cellIdxs'][b, ...] = all_data[:, 0]
		dataset[i]['originalIndex'][b, ...] = all_data[:, 1]
		dataset[i]['gridShape'][b, ...] = gridShape
		dataset[i]['locs'][b, ...] = all_data[:, 2:6]
		dataset[i]['data'][b, ...] = all_data[:, 6:9].transpose(0,1)
		dataset[i]['density'][b, ...] = all_data[:, 9:]
		k = 0
		for j in range(int(all_data[:, 0].max())):
			dataset[i]['cellStart'][b, j] = k
			while all_data[k, 0] == j:
				k += 1
			dataset[i]['cellEnd'][b, j] = k
	print("")
# import cutil
# cutil.keyboard()


results = np.zeros((len(KERNEL_SIZES), len(CHANNEL_SIZES)), dtype=np.float)
for j, channels in enumerate(CHANNEL_SIZES):
	print("Adjusting data dimensions to %d..." % channels)
	for d in dataset:
		dd = d['data'].numpy()
		s = list(dd.shape)
		s[1] = channels
		d['data'] = torch.from_numpy(np.resize(dd, s))
	for i, kernel_size in enumerate(KERNEL_SIZES):
		print("\tRunning tests for kernel size %s..." % str(kernel_size))
		conv = spn.ConvSP(channels, channels, dataset[0]['locs'].size()[-1] - 1,
			kernel_size, 0.05, 0.1)
		conv = cudafy(conv)
		[p.data.normal_(0, 1) for p in conv.parameters()]
		t = 0
		for k, d in enumerate(dataset):
			print("\t\tTesting batch %d..." % k)
			locs = Variable(cudafy(d['locs']))
			data = Variable(cudafy(d['data']))
			density = Variable(cudafy(d['density']))
			cellIdxs = Variable(cudafy(d['cellIdxs']))
			originalIndex = Variable(cudafy(d['originalIndex']))
			cellStart = Variable(cudafy(d['cellStart']))
			cellEnd = Variable(cudafy(d['cellEnd']))
			gridShape = Variable(cudafy(d['gridShape']))
			start = time.time()
			conv(locs, data, density, cellIdxs, originalIndex, cellStart, cellEnd, gridShape)
			t += time.time() - start
		t /= len(dataset)
		print("\tAveraged %f/batch." % t)
		results[i, j] = t

headers = ["Channels"] + [str(x) for x in KERNEL_SIZES]
hlen = max([len(x) for x in headers])
print(' '.join("%*s"%(hlen, x) for x in headers))
for j, channels in enumerate(CHANNEL_SIZES):
	line = ["%*d" % (hlen, channels)]
	for i in range(len(KERNEL_SIZES)):
		line.append("%*f" % (hlen, results[i, j]))
	print(' '.join(line))

