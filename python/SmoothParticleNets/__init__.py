
from os.path import dirname, basename, isfile
import glob
import sys
sys.path.append(dirname(__file__))
modules = glob.glob(dirname(__file__)+"/*.py")
__all__ = [basename(f)[:-3] for f in modules if isfile(f)]
for f in modules:
    if isfile(f) and "__init__" not in f and "install" not in f:
        exec('from %s import *' % basename(f)[:-3])
