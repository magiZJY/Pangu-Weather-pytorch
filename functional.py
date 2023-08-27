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

def TransposeDimensions(tensor: Tensor, *dims: int):
    return tensor.permute(*dims)

class MLP(torch.nn.Module):
    def __init__(self, in_features: int, drop: float=0.):
        """
        A simple multi-layer perceptron. {in_features} = {out_features} = {hidden_features}.
        """
        super().__init__()
        out_features = in_features
        hidden_features = in_features
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_features, out_features)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.fc2(x)
        x=self.drop(x)
        return x