"""
dnn.np - deep MLP on CIFAR-10

base MLP module extented with:
    batchnorm
    dropout
    GELU (with ReLU)
    decoupled-weight-decay AdamW optim
    cross-entropy backward

* everything is derived manually, no autograd.

all gradients are verified against finite differences 
in 'check_grads()' before training.
"""

from __future__ import annotations

import os
import pickle
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Iterator, Union, Dict, Any

import numpy as np


## Core primitiives

@dataclass
class Param:
    """A learnable tensor bundled withs its accumulated gradient."""
    value: np.ndarray
    grad: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.grad = np.zeros_like(self.value)

    def zero_grad(self) -> None:
        seld.grad.fill(0.0)

class Layer:
    # only meaningful for batchnorm/dropout
    training: bool = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> List[Param]:
        return []