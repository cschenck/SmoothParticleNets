# SmoothParticleNets

Smooth Particle Networks (SmoothParticleNets or SPNets) is a set of custom PyTorch layers to facilitate computation with unordered particle sets.
They were created for the purpose of enabling particle-based fluid dynamics inside a deep network, but the layers can be used for other purposes.
Broadly, the layers enable computing particle-particle interactions, particle-object interactions, and projections onto and out of a camera image.
The interface to this library is in Python.
This library contains 6 layers, listed below.
Note that this library provides only the basic functionality and no additional utilities, e.g., the library does not include a particle visualizer and the library does not include a tool for processing 3D object mesh files into signed distance fields.

## Layers

Below is the list of each layer contained in this library.
Clicking on the layer's name will take you to a description of what that layer does and how to use it.

* [ConvSP](https://cschenck.github.io/SmoothParticleNets/docs/convsp)
* [ConvSDF](https://cschenck.github.io/SmoothParticleNets/docs/convsdf)
* [ImageProjection](https://cschenck.github.io/SmoothParticleNets/docs/imageprojection)
* [ParticleProjection](https://cschenck.github.io/SmoothParticleNets/docs/particleprojection)
* [ParticleCollision](https://cschenck.github.io/SmoothParticleNets/docs/particlecollision)
* [ReorderData](https://cschenck.github.io/SmoothParticleNets/docs/reorderdata)

## Requirements

This library only requires PyTorch as a dependency. 
The current version of the library has been tested to work with PyTorch 0.4.1.
Furthermore, this library only supports Python 3, and does not support Python 2.

Note that this library was developed only under linux and may or may not run directly without modification on other platforms.
Specifically, this library is confirmed to work on Ubuntu 18.04 with PyTorch 0.4.1, Cuda 10.0, and the 410 Nvidia drivers (although that should not matter).

## Installation

To install this library, download the source from github.
Once downloaded, enter the root directory of the source and run
```bash
sudo python3 setup.py install
```

Once installed, in Python you should be able to call 'import SmoothParticleNets', which will import the library.

## Citation

In published works please cite this as
> C. Schenck and D. Fox, "SPNets: Differentiable Fluid Dynamics for Deep Neural Networks," in *Proceedings of the Second Conference on Robot Learning (CoRL),* Zurich, Switzerland, 2018.

```bibtex
@inproceedings{spnets2018,
  title={SPNets: Differentiable Fluid Dynamics for Deep Neural Networks},
  author={Schenck, C. and Fox, D.},
  booktitle={Proceedings of the Second Conference on Robot Learning (CoRL)},
  year={2018},
  address={Zurich, Switzerland}
}
```
