
import os
import subprocess as sp
import sys
from torch.utils.ffi import create_extension

CUDA_SRCS = ['gpu_kernels.cu']
HEADERS = ['spn.h']
SRCS = ['fgrid.c', 'particles2grid.c']

root_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(root_dir, "lib")
src_dir = os.path.join(root_dir, "src")
py_dir = os.path.join(root_dir, "python", "SmoothParticleNets")

if not os.path.exists(lib_dir):
	os.mkdir(lib_dir)

for cuda_src in CUDA_SRCS:
	if sp.call(("nvcc -c -o %s %s -x cu -Xcompiler -fPIC -arch=sm_52" 
		% (os.path.join(lib_dir, cuda_src + ".o"), os.path.join(src_dir, cuda_src))).split()):
		sys.exit()

cwd = os.getcwd()
os.chdir(py_dir)
ffi = create_extension(
	name='_ext',
	headers=[os.path.join(src_dir, h) for h in HEADERS],
	sources=[os.path.join(src_dir, s) for s in SRCS],
	define_macros=[('WITH_CUDA', None)],
	with_cuda=True,
	extra_objects=[os.path.join(lib_dir, o + '.o') for o in CUDA_SRCS]
)
ffi.build()
os.chdir(cwd)