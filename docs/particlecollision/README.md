# ParticleCollision

[SmoothParticleNets](https://cschenck.github.io/SmoothParticleNets)

## Description

The ParticleCollision layer pre-computes neighbor lists (i.e., "colliding" particles) for each given particle.
That is, given a list of particle positions and a fixed radius, this layer returns a short list for each particle with the index of all other particles that are within that radius of it.
To do this, internally the ParticleCollision layer creates a hashgrid and performs lookups based on that grid.
The resulting neighbor list is designed to be used by the ConvSP layer to compute particle-particle interactions.

An important operation that this layer does alongside computing collisions is to reorder the particle list.
The reordering places particles falling in the same grid cell in the hash grid next to each other in memory.
By doing so, cache hits are increased dramatically during the computation of particle-particle interactions in ConvSP, resulting in a large speedup.
Due to this reordering, the returned list of colliding neighbor indices are indices in the *reordered* list, not in the original.
The standard use of this layer is to compute collisions, make as many calls to ConvSP as are desired, then use the ReorderData layer to return the particle list to its original order.
It is important to emphasize that reordering the data according to the hash grid is critical for perfomance of the ConvSP layer.

ParticleCollision is implemented as a subclass of torch.nn.Module.
This allows it to be used in the same manner as any other PyTorch layer (e.g., conv2d).
There are no gradients to compute for this layer, so it simply passes them through when calling backward.

## Example

Assume *locs* is a BxNxD tensor containing the locations of N D-dimensional particles across B batches and vel is a same size tensor containing the particle's velocities.
```python
coll = ParticleCollision(ndim, radius)
# PartileCollision reorders locs and vel.
locs, vel, idxs, neighbors = coll(locs, vel)
```


## Documentation

ParticleCollision provides two functions: a constructor and forward.
Forward is called by calling the layer object itself (in the same manner as any standard PyTorch layer).

* ### ParticleCollision(ndim, radius, max_grid_dim=96, max_collisions=128, include_self=True):
    * Arguments
        * **ndim**[int]: The dimensionality of the particle's coordinate space.
        * **radius**[float]: The maximum distance a particle can be from another and still be colliding.
        * **max_grid_dims**[int]: (optional) The maximum size of the hash grid in any dimension. This is useful for limiting memory consumpation in cases where the particles are very spread out relative to the collision radius. Particles that don't fall in the hash grid are placed in the cell closest to them. 
        * **max_collisions**[int]: (optional) The maximum number of neighbors to return. The returned neighbor list for each particle will always be this length (although not necessarily entirely filled in), so selecting this parameter is a balance between memory consumption and ensuring all colliding particles are included.
        * **include_self**[boolean]: (optional) If True, the particle will be in its own list of neighbors. If False it will not be.

* ### forward(idxs, locs, data=None, qlocs=None):
    * Arguments
        * **locs**[BxNxD torch.autograd.Variable]: The batched list of particle locations. D must match the ndim argument to the constructor.
        * **data**[BxNxK torch.autograd.Variable]: (optional) Additional data associated with each particle. This data is not used during the forward call, however since the locs are reordered, any data associated with each particle must also be reordered. Technically this could also be accomplished instead by calling the ReorderData layer on the data after calling forward, but doing so here helps to prevent bugs when calling ConvSP with reordered locs but non-reordered data.
        * **qlocs**[BxMxD torch.autograd.Variable]: (optional) In the case where it is desired to compute collisions between two different particle sets, this is the second set. Rather than returning the neighbor list for particles in locs, if this argument is passed, the returned neighbor list is a list for each particle in qlocs of the indices of particles in locs (after reordering) that it collides with.
    * Returns
        * **locs**[BxNxD torch.autograd.Variable]: The reordered list of particle positions.
        * **data**[BxNxK torch.autograd.Variable]: (optional) If data was passed as an input, then the data reordered is returned.
        * **idxs**[BxNxD torch.autograd.Variable]: The index list for the reordered particle list. Each index value indicates where the original index of that particle in the original locs, i.e., idxs[b, i] = j where i is the new index of the particle after reordering and j is its original index (b being the batch).
        * **neighbors**[Bx(N/M)xC torch.autograd.Variable]: The neighbor list for each particle. If qlocs was passed as an argument, then it is the neighbors of each particle in qlocs instead of locs. Each value indicates the index in locs (after reordering) of the neighboring particle. C is the value of max_collisions as passed to the constructor. Note that not all particles will have max_collisions neighbors. In that event, the values in each particle's list are filled sequentially, with unfilled values in the list being set to -1.
