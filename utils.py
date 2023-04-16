"""
PFE Boulet Olgiati
"""

import numpy as np


def identity(x):
    return x

def tanh(x) : 
  return np.tanh(x)

def forward(self, s, alpha) : 
    return (1 - alpha) * np.tanh(s)

def correct_dimensions(s, targetlength):   # https://github.com/cknd/pyESN/blob/master/pyESN.py
    """checks the dimensionality of some numeric argument s, broadcasts it
       to the specified length if possible.
    Args:
        s: None, scalar or 1D array
        targetlength: expected length of s
    Returns:
        None if s is None, else numpy vector of length targetlength
    """
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s