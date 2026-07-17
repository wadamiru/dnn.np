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
contain data_batch_1..5 and test_batch). If no such directory is found,
this script falls back to a small synthetic dataset of the same shape so
the pipeline (and gradient checks) can still be exercised end-to-end.
"""

