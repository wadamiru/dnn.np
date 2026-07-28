"""
dnn.np - deep MLP on CIFAR-10

base MLP module extented with:
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

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.W.grad += self._x.T @ dout
        self.b.grad += dout.sum(axis=0)
        return dout @ self.W.val.T      # dx

    def params(self) -> List[Params]:
        return [self.W, self.b]

class GELU(Layer):
    """
    GELU, tanh approx. (as used in GPT-2/BERT):
        y = 0.5 * x * (1 + tanh(c * (x + 0.044715 * x^3))), c = sqrt(2/pi)

    For backward pass:
    Let u = c * (x + 0.044715 * x^3), t = tanh(u):
        y = 0.5 * x * (1 + t)
    """

    _c = np.float32(np.sqrt(2/np.pi))

    def forward(self, x: np.ndarray) -> np.ndarray:
        u = self._c * (x + 0.044715 * x**3)
        t = np.tanh(u)
        self._x, self._t = x, t
        return 0.5 * x * (1 + t)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, t, c = self._x, self._t, self._c
        du = c * (1.0 + 0.134145 * x**2)
        dy = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * du
        return dout * dy

class ReLU(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self._mask