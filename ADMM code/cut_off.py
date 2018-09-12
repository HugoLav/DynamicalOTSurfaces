"""
Definition of a cut off function
"""

# Mathematical functions
from math import *

def f(x, sigma):
    """Define a cutoff function = 1 if x <= 0, = 0 if x>=sigma, smooth transition in between
    """
    x /= sigma
    if x <= 0.:
        return 1.
    elif x >= 1.:
        return 0.
    else:
        return (x - 1.) ** 2 * (x + 1) ** 2
