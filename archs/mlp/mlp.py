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
from typing import List, Tuple, Iterator, Union, Dict, Any

import numpy as np


## Core primitiives

@dataclass
class Param:
    """A learnable tensor bundled withs its accumulated gradient."""
    val: np.ndarray
    grad: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.grad = np.zeros_like(self.val)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)

class Layer:
    # only meaningful for batchnorm/dropout
    training: bool = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> List[Param]:
        return []

class Linear(Layer):
    """y = x @ W + b linear layer."""

    def __init__(self, nin: int, nout: int, init: str="he"):
        std = np.sqrt(2.0 / nin) if init == "he" else np.sqrt(1.0 / nin)
        W = (np.random.randn(nin, nout) * std).astype(np.float32)
        b = np.zeros(nout, dtype=np.float32)
        self.W = Param(W)
        self.b = Param(b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W.val + self.b.val

    def backward(self, )