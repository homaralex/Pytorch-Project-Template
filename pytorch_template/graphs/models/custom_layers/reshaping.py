import torch.nn as nn


class View(nn.Module):
    """Based on https://github.com/pytorch/vision/issues/720#issuecomment-477699115"""
    def __init__(self, shape):
        super(View, self).__init__()

        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
