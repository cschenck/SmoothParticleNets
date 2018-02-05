
import argparse
import os 
import subprocess as sp
import sys
from torch.utils.ffi import create_extension

CUDA_SRCS = ['gpu_kernels.cu']
CUDA_HEADERS = ['cuda_spn.h']
CPU_ONLY_HEADERS = ['non_cuda_spn.h']
HEADERS = ['spn.h']
SRCS = ['cuda_layer_funcs.c', 'cpu_layer_funcs.c']

parser = argparse.ArgumentParser()
parser.add_argument('--with_cuda', action="store_true", default=False)
parser.add_argument('--debug', action="store_true", default=False)
args = parser.parse_args()

root_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(root_dir, "lib")
src_dir = os.path.join(root_dir, "src")
test_dir = os.path.join(root_dir, "test")
py_dir = os.path.join(root_dir, "python", "SmoothParticleNets")

if not os.path.exists(lib_dir):
	os.mkdir(lib_dir)

macros = []
objects = []
pytest_args = {}
if args.with_cuda:
	for cuda_src in CUDA_SRCS:
		if sp.call(("nvcc -c -o %s %s -x cu -Xcompiler -fPIC -arch=sm_52 %s" 
			% (os.path.join(lib_dir, cuda_src + ".o"), os.path.join(src_dir, cuda_src),
				"-g -G" if args.debug else "")).split()):
			sys.exit(-1)
	macros.append(('WITH_CUDA', None))
	objects += [os.path.join(lib_dir, o + '.o') for o in CUDA_SRCS]
	HEADERS += CUDA_HEADERS
	pytest_args['with_cuda'] = True
else:
	HEADERS += CPU_ONLY_HEADERS
	pytest_args['with_cuda'] = False

cwd = os.getcwd()
os.chdir(py_dir)
ffi = create_extension(
	name='_ext',
	headers=[os.path.join(src_dir, h) for h in HEADERS],
	sources=[os.path.join(src_dir, s) for s in SRCS],
	define_macros=macros,
	with_cuda=args.with_cuda,
	extra_objects=objects
)
ffi.build()
os.chdir(cwd)

fp = open(os.path.join(test_dir, "pytest_args.py"), "w")
for k, v in pytest_args.items():
	if isinstance(v, str):
		v = "'" + v + "'"
	fp.write("%s = %s\n" % (k, str(v)))
fp.close()