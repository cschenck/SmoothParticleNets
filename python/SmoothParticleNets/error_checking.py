
import numbers
import numpy as np

import torch

def check_nans(v, name):
    if (v != v).data.any():
        raise ValueError("Found NaNs in %s" % name)

def throws_exception(exception_type, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
        return False
    except exception_type:
        return True

def check_conditions(v, name, *conditions):
    for condition in conditions:
        if not eval(condition % "v"):
            raise ValueError(("%s must meet the following condition: " + condition) 
            	% (name, name))
    return v

def make_list(l, length, name, *conditions):
    if throws_exception(TypeError, list, l):
        l = [l]*length
    else:
        l = list(l)
    if len(l) != length:
        raise ValueError("%s must be a list of length %d." % (name, length))
    for i, ll in enumerate(l):
        l[i] = check_conditions(ll, name, *conditions)
    return l

def check_tensor_dims(t, name, dims):
    s = t.size()
    if len(s) != len(dims):
        raise ValueError("%s must be a %d-dimensional tensor." % (name, len(dims)))
    for i in range(len(dims)):
        if dims[i] >= 0 and s[i] != dims[i]:
            raise ValueError("The %dth dimension of %s must have size %d, not %d." 
            	% (i, name, dims[i], s[i]))

def list2tensor(l):
    return torch.from_numpy(np.array(l, dtype=np.float32))