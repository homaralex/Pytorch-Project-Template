"""
Mnist Main agent, as mentioned in the tutorial
"""

import torch.nn.functional as F

from agents.base import BaseTrainAgent
from graphs.models.mnist import Mnist
from datasets.mnist import MnistDataLoader


class MnistAgent(BaseTrainAgent):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # set agent name
        self.agent_name = 'MNIST'
        # define loss
        self.loss_fn = F.nll_loss

    def _init_model(self):
        self.model = Mnist()

    def _init_data_loader(self):
        self.data_loader = MnistDataLoader(config=self.config)
