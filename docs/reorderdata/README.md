# ReorderData

[SmoothParticleNets](https://cschenck.github.io/SmoothParticleNets)

## Description

The ReorderData layer is fairly simple.
The layer reorders a given tensor based on a tensor containing the indices for the data in the first tensor.
More formally, assume that DATA is a BxNxD tensor containing N D-dimensional data points (e.g., XYZ particle locations) over B batches.
Let IDXS be a BxN tensor, where each IDXS[i, :] contains the numbers 0 to N-1 in some arbitrary order.
This layer then returns DATA where the second dimension has been rearranged according to IDXS.
This is equivalent to 
```python
DATA[i, :, :] = DATA[i, IDXS[i, :], :]
```
in PyTorch syntax, however this layer is specialized for this specific kind of indexing resulting in a faster implementation.
This layer is designed as a helper layer for the ParticleCollision layer.

ReorderData is implemented as a subclass of torch.nn.Module.
This allows it to be used in the same manner as any other PyTorch layer (e.g., conv2d).
Additionally, this layer computes graidents, so it can be used in a backward pass.

## Example

Assume *locs* is a BxNxD tensor containing the locations of N D-dimensional particles across B batches and *vel* is a same size tensor containing the particles' velocities.
```python
# ReorderData is most commonly used in conjunction with ParticleCollision.
coll = ParticleCollision(ndim, radius)
# Set reverse=True. ParticleCollision calls ReorderData internally, so we want to undo that reordering when we're done.
reorder = ReorderData(reverse=True)
# PartileCollision reorders locs and vel.
locs, vel, idxs, neighbors = coll(locs, vel)
# Perform desired operations with locs, vel, neighbors...
# When we're done, return locs and vel to their original order using ReorderData.
locs, vel = reorder(idxs, locs, vel)
```


## Documentation

ReorderData provides two functions: a constructor and forward.
Forward is called by calling the layer object itself (in the same manner as any standard PyTorch layer).

* ### ReorderData(reverse=True):
    * Arguments
        * **reverse**[boolean]: (optional) When False, behaves as normal, using the given indices to reorder the data. When True, this layer assumes that the given data was already reordered according to the given indices, and so reverses that process and retursn the data to the original order.

* ### forward(idxs, locs, data=None):
    * Arguments
        * **idxs**[BxN torch.autograd.Variable]: The list of indices to redorder the input by.
        * **locs**[BxNxD torch.autograd.Variable]: The main data to be reordered. It is called *locs* because ReorderData is primarily a helper for ParticleCollision, which reorders the locations of the particles.
        * **data**[BxNxK torch.autograd.Variable]: (optional) Additional data to reorder alongside locs. Calling forward with both locs and data is equivalent to calling it twice in a row with each individually. This argument is provided as a convenience.
    * Returns
        * **locs**[BxNxD torch.autograd.Variable]: A new tensor with the same values as in the locs argument reordered based in idxs.
        * **data**[BxNxK torch.autograd.Variable]: (optional) If the data argument is passed, then forward will return a pair of tensors, where the second has the same values as data but reordered according to idxs.