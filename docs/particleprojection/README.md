# ParticleProjection

[SmoothParticleNets](https://cschenck.github.io/SmoothParticleNets)

## Description

The ParticleProjection layer is designed to allow comparison of the particle state with a camera image.
It does this by projecting the particles onto a virtual camera image, which can then be compared to other camera images as desired.
Each particle is projected onto the virtual image as a small Gaussian, which allows for smooth gradients with respect to the particle positions or camera pose.
The layer computes the image coordinate of a given particle location using the pinhole camera model, not taking into account any distortions, e.g., radial distortion.
ParticleProjection currently only supports 3D particle locations.

ParticleProjection is implemented as a subclass of torch.nn.Module.
This allows it to be used in the same manner as any other PyTorch layer (e.g., conv2d).
ParticleProjection can compute gradients with respect to the camera or particle poses, and is implemented with Cuda support for efficient computation.

## Example

Assume *locs* is a BxNxD tensor containing the locations of N D-dimensional particles across B batches.
```python
# First create the ParticleProjection layer.
proj = ParticleProjection(camera_fl=540, camera_size=(480, 640), filter_std=5.0, filter_scale=10.0)
# Setup the camera pose.
camera_pose = torch.Tensor([0.0, 0.0, 0.0])
camera_rotation = torch.Tensor([0.0, 0.0, 0.0, 1.0])
image = proj(locs, camera_pose, camera_rotation)
```


## Documentation

ParticleProjection provides two functions: a constructor and forward.
Forward is called by calling the layer object itself (in the same manner as any standard PyTorch layer).

* ### ParticleProjection(camera_fl, camera_size, filter_std, filter_scale):
    * Arguments
        * **camera_fl**[float]: The focal length of the camera.
        * **camera_size**[tuple]: A tuple of the camera image height and width (in that order) in pixels.
        * **filter_std**[float]: The standard deviation (in pixels) of the Gaussian for each particle. The Gaussian will be added to all pixels within 2x of this to the particle's image coordinate.
        * **filter_scale**[float]: All values added to a pixel will be multiplied by this to allow control of the intensity of the Gaussians for each particle. This is equivalent to multiplying the output image by this value after the fact.

* ### forward(locs, camera_pose, camera_rot, depth_mask=None):
    * Arguments
        * **locs**[BxNx3 torch.autograd.Variable]: The batched list of particle locations. Only 3D particle loations are supported.
        * **camera_pose**[Bx3 torch.autograd.Variable]: The camera translation in the environment.
        * **camera_rot**[Bx4 torch.autograd.Variable]: The camera rotation in the environment, represented as a quaternion in xyzw format.
        * **depth_mask**[BxHxW torch.autograd.Variable]: (optional) If passed, this is used to mask particles that are obscured by obstructions in the environment. If the depth of a pixel is less than the depth of the particle, the particle's contribution to that pixel is not added. H and W must match the camera image height and width passed to the constructor. 
    * Returns
        * **image**[BxHxW torch.autograd.Variable]: The projected image. Particles appear as small Gaussians, and where particles overlap the Gaussians are added together.

