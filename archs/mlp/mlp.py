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

CIFAR-10 data
-------------
Download the "CIFAR-10 python version" from
https://www.cs.toronto.edu/~kriz/cifar.html, extract it, and point
`--data-dir` at the extracted `cifar-10-batches-py/` folder (it should
contain data_batch_1..5 and test_batch). 

If no such directory is found,
this script falls back to a small synthetic dataset of the same shape so
the pipeline (and gradient checks) can still be exercised end-to-end.
"""

from __future__ import annotations

import os
import pickle
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Iterator, Union, Dict, Any

import numpy as np


# Core primitiives
# ----------------

@dataclass
class Parameter:
    """A learnable tensor bundled withs its accumulated gradient."""
    value: np.ndarray
    grad: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.grad = np.zeros_like(self.value)

    def zero_grad(self) -> None:
        seld.grad.fill(0.0)