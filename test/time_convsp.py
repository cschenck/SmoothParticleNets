

import os
import sys
# Add path to python source to path.
sys.path.append(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import SmoothParticleNets as spn
import argparse
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

CHANNEL_SIZES = range(1, 33)
RADIUS = 0.04
DILATION = 0.01

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', required=True, action="store", type=str)
args = parser.parse_args()



print("Loading dataset...")
dataset = pickle.load(open(args.datapath, "rb"))

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
			kernel_size, RADIUS, DILATION)
		conv = conv.cuda()
		[p.data.normal_(0, 1) for p in conv.parameters()]
		t = 0
		for k, d in enumerate(dataset):
			print("\t\tTesting batch %d..." % k)
			locs = Variable(d['locs'].cuda())
			data = Variable(d['data'].cuda())
			density = Variable(d['density'].cuda())
			start = time.time()
			conv(locs, data, density)
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

