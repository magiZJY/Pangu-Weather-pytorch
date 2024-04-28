import torch
import torch.nn as nn

import torch.nn.functional as F

def Pad3D(x, padding):
    """
    Apply 3D padding to a 5D tensor.

    Parameters:
    - x: A 5D tensor of shape [batch_size, channels, depth, height, width].
    - padding: A tuple of 6 integers (padLeft, padRight, padTop, padBottom, padFront, padBack)
               specifying the amount of padding to add.

    Returns:
    - A 5D tensor with padding applied.
    """
    return F.pad(x, padding, mode='constant', value=0)


def Pad2D(x, padding):
    """
    Apply 2D padding to a 4D tensor.

    Parameters:
    - x: A 4D tensor of shape [batch_size, channels, height, width].
    - padding: A tuple of 4 integers (padLeft, padRight, padTop, padBottom)
               specifying the amount of padding to add.

    Returns:
    - A 4D tensor with padding applied.
    """
    return F.pad(x, padding, mode='constant', value=0)

# Example usage
x = torch.randn(1, 1, 10, 10)  # Example tensor
padding = (1, 1, 1, 1)  # Add a padding of 1 unit around each side in the height and width dimensions

def gen_mask(x):

    return

def no_mask(x):
    return