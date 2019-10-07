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
        self.logger.info(f'Model architecture:\n{self.model}\nKeras-style summary:\n{self.model.summary()}')

    def _init_data_loader(self):
        self.data_loader = MnistDataLoader()
