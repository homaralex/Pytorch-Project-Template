"""
Mnist Main agent, as mentioned in the tutorial
"""

import torch.nn.functional as F

from pytorch_template.agents.base import BaseTrainAgent
from pytorch_template.datasets.mnist import MnistDataLoader
from pytorch_template.graphs.models.mnist import Mnist


class MnistAgent(BaseTrainAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.agent_name = 'MNIST'
        self.loss_fn = F.nll_loss

    def _init_model(self):
        self.model = Mnist()

    def _init_data_loader(self):
        self.data_loader = MnistDataLoader()
