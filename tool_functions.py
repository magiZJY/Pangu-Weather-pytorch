import torch
from torch import Tensor
import torch.nn.init as init
import torch.nn.functional as F

def roll3D(input_tensor, shifts, dims):
    return torch.roll(input_tensor, shifts, dims)


def pad3D(input_tensor, pad):
    return F.pad(input_tensor, pad)


def pad2D(input_tensor, pad):
    return F.pad(input_tensor, pad)


def Crop3D(
    input_tensor, start_dim1, start_dim2, start_dim3, size_dim1, size_dim2, size_dim3
):
    return input_tensor[
        start_dim1 : start_dim1 + size_dim1,
        start_dim2 : start_dim2 + size_dim2,
        start_dim3 : start_dim3 + size_dim3,
    ]


def Crop2D(input_tensor, start_dim1, start_dim2, size_dim1, size_dim2):
    return input_tensor[
        start_dim1 : start_dim1 + size_dim1, start_dim2 : start_dim2 + size_dim2
    ]

def TruncatedNormalInit(tensor: Tensor, mean: float = 0., std: float = 1., a: float = -2., b: float = 2.) -> Tensor:
    return init._no_grad_trunc_normal_(tensor, mean, std, a, b)

def Backward(tensor: Tensor):
    tensor.backward()

def UpdateModelParametersWithAdam(AdamOptimizer: torch.optim.Adam):
    AdamOptimizer.step()