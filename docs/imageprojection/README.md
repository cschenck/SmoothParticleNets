# ImageProjection

[SmoothParticleNets](https://cschenck.github.io/SmoothParticleNets)

## Description

The ImageProjection layer projects an image feature map onto a set of particles in the view frame of the camera.
That is, given an image of C channels, it first projects each particle onto the image using given camera intrinsics (focal length, etc.) and extrinsics (pose).
Then it uses bilinear interpolation between the 4 adjacent pixels to generate a feature vector for the given particle.
The output is a C-length feature vector for each particle.
The ImageProjection layer currently only supports 3D coordinate spaces.

ImageProjection is implemented as a subclass of torch.nn.Module.
This allows it to be used in the same manner as any other PyTorch layer (e.g., conv2d).
ImageProjection can compute gradients with respect to the camera or particle poses and the image features, and is implemented with Cuda support for efficient computation.

## Example

Assume *locs* is a BxNxD tensor containing the locations of N D-dimensional particles across B batches and image is a [BxHxWxC] feature image.
```python
# First create the ParticleProjection layer.
proj = ImageProjection(camera_fl=540)
# Setup the camera pose.
camera_pose = torch.Tensor([0.0, 0.0, 0.0])
camera_rotation = torch.Tensor([0.0, 0.0, 0.0, 1.0])
new_data = proj(locs, image, camera_pose, camera_rotation)
```


## Documentation

ImageProjection provides two functions: a constructor and forward.
Forward is called by calling the layer object itself (in the same manner as any standard PyTorch layer).

* ### ImageProjection(camera_fl):
    * Arguments
        * **camera_fl**[float]: The focal length of the camera.

* ### forward(locs, image, camera_pose, camera_rot, depth_mask=None):
    * Arguments
        * **locs**[BxNx3 torch.autograd.Variable]: The batched list of particle locations. Only 3D particle loations are supported.
        * **image**[BxHxWxC torch.autograd.Variable]: The image to project onto the particles. H and W are the height and width, respectively, and C is the number of channels.
        * **camera_pose**[Bx3 torch.autograd.Variable]: The camera translation in the environment.
        * **camera_rot**[Bx4 torch.autograd.Variable]: The camera rotation in the environment, represented as a quaternion in xyzw format.
        * **depth_mask**[BxHxW torch.autograd.Variable]: (optional) If passed, this is used to mask particles that are obscured by obstructions in the environment. If the depth of a pixel is less than the depth of the particle, nothing is projected onto that particle. 
    * Returns
        * **new_data**[BxNxC torch.autograd.Variable]: The set of features for each particle after projecting the image features onto them.


