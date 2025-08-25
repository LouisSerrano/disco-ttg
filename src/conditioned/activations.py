"""Local activation registry to replace pdearena dependency"""

import torch.nn as nn

ACTIVATION_REGISTRY = {
    "gelu": nn.GELU(),
    "relu": nn.ReLU(), 
    "silu": nn.SiLU(),
    "swish": nn.SiLU(),  # SiLU is also known as Swish
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(),
    "elu": nn.ELU(),
}