#!/usr/bin/env python

import argparse
import os
import shutil
import site
import sys

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-l', '--local', dest='local', action="store_true",
    help='Rather than copying the library files over to your python package directory,'
         ' leave them here in the source and add the source directory to Python\'s search'
         ' path. NOTE: Moving or changing the source files here will be change your '
         'SmoothParticleNets library. This may be desirable if you are actively developing'
         ' it. Otherwise you may want to use the --global flag instead.')
group.add_argument('-g', '--global', dest='local', action='store_false',
    help='Copy the library files to your Python package directory. Changes made here to '
         'source will not be reflected in your SmoothParticleNets library.')
parser.add_argument('--package_path', action="store", type=str, default=None, 
    help='If specified, this is the Python package directory the SmoothParticleNets '
         'library will be installed to. If not specified, the site Python package will be'
         ' used to find the Python package path.')
group.add_argument('-r', '--remove', action='store_true', default=False, 
        help='Instead of installing, remove SmoothParticleNets.')
args = parser.parse_args()

if args.package_path is None:
    packdir = site.getsitepackages()[0]
    print("Package path not specified, using %s." % packdir)
else:
    packdir = args.package_path

try:

    if args.remove:
        if "SmoothParticleNets" in os.listdir(packdir):
            print("Removing global copy of SmoothParticleNets...")
            shutil.rmtree(os.path.join(packdir, "SmoothParticleNets"))
        if "import_SmoothParticleNets.pth" in os.listdir(packdir):
            print("Removing reference to local copy of SmoothParticleNets...")
            os.remove(os.path.join(packdir, "import_SmoothParticleNets.pth"))
        print("SmoothParticleNets removed.")
        sys.exit()

    if "SmoothParticleNets" in os.listdir(packdir):
        print("Found SmoothParticleNets library already installed in your Python package directory."
              " It must be removed to continue installation.")
        s = raw_input("Remove existing SmoothParticleNets (Y/n)? ")
        if s.lower() != 'y':
            print("Aborting install.")
            sys.exit()
        shutil.rmtree(os.path.join(packdir, "SmoothParticleNets"))


    if args.local:
        print("Installing reference to local source directory...")
        f = open(os.path.join(packdir, "import_SmoothParticleNets.pth"), "w")
        f.write(os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                "python"))
        f.flush()
        f.close()
        print("Installed.")
    else:
        if "import_SmoothParticleNets.pth" in os.listdir(packdir):
            print("Found reference to local install in your Python package directory. "
                  "It must be removed to complete installation.")
            s = raw_input("Remove reference to local SmoothParticleNets (Y/n)? ")
            if s.lower() != 'y':
                print("Aborting install.")
                sys.exit()
            os.remove(os.path.join(packdir, "import_SmoothParticleNets.pth"))
        print("Copying SmoothParticleNets library to Python package directory...")
        shutil.copytree(os.path.join(os.path.dirname(os.path.realpath(__file__)), 
          "python", "SmoothParticleNets"), os.path.join(packdir, "SmoothParticleNets"))
        print("Installed.")

except IOError:
    print("ERROR: Please make sure you are root or have write permissions in %s." % packdir)
    print("Run \'sudo ./install.py\' to install correctly.")
