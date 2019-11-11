"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import logging
from contextlib import ExitStack

import gin
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from pytorch_template.graphs.optimizers import sgd
from pytorch_template.utils import dirs as module_dirs
from pytorch_template.utils.devices import configure_device
from pytorch_template.utils.misc import is_debug_mode


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self):
        self.logger = logging.getLogger("Agent")
        self.agent_name = 'Base'

    def _get_state_dict(self):
        return {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'torch_random_state': torch.get_rng_state(),
            'numpy_random_state': np.random.get_state(),
        }

    def _load_state_dict(self, state_dict):
        self.current_epoch = state_dict['epoch'] + 1
        self.current_iteration = state_dict['iteration']

        torch.set_rng_state(state_dict['torch_random_state'].cpu())
        np.random.set_state(state_dict['numpy_random_state'])

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, is_best=False):
        """
        Checkpoint saver
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        raise NotImplementedError

    @property
    def debug(self):
        return is_debug_mode()


@gin.configurable
class BaseTrainAgent(BaseAgent):
    def __init__(
            self,
            max_epoch,
            log_interval=10,
            checkpoint_path=None,
    ):
        super().__init__()

        self.agent_name = 'BaseTrainAgent'
        self.max_epoch = max_epoch
        self.log_interval = log_interval

        self._init_counters()
        self._init_model()
        self._init_optimizer()
        self._init_data_loader()
        self._init_device()
        self._init_tboard_logging()

        self.load_checkpoint(checkpoint_path)

    def _init_counters(self):
        self.current_epoch = 0
        self.current_iteration = 0

    def _init_model(self):
        raise NotImplementedError()

    def _init_optimizer(self):
        self.optimizer = sgd(params=self.model.parameters())

    def _init_data_loader(self):
        raise NotImplementedError()

    def _init_device(self):
        self.device = configure_device()
        self._move_to_device()

    def _move_to_device(self):
        self.model = self.model.to(self.device)

    def _init_tboard_logging(self):
        self.summary_writer = SummaryWriter(
            log_dir=module_dirs.get_current_tboard_dir(),
            comment=self.agent_name,
        )
        # add config string to summary
        config_str = gin.config_str()
        # see https://stackoverflow.com/questions/45016458/tensorflow-tf-summary-text-and-linebreaks
        config_str = config_str.replace('\n', '  \n')
        self.summary_writer.add_text(tag='gin_config', text_string=config_str)

    def _get_state_dict(self):
        state_dict = super()._get_state_dict()

        state_dict.update({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        })

        return state_dict

    def _load_state_dict(self, state_dict):
        super()._load_state_dict(state_dict)

        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    @property
    def checkpoints_dir(self):
        return module_dirs.get_current_checkpoints_dir()

    @property
    def experiments_dir(self):
        # checkpoint_dir is of structure .../exp_name/datetime_str/
        # and we will look for the checkpoint file under a different timestamp
        return self.checkpoints_dir.parent

    def load_checkpoint(self, file_name=None):
        """
        Load model from the latest checkpoint - if not specified start from scratch.
        :param file_name: name of the checkpoint file
        :return:
        """
        if file_name is not None and len(file_name) > 0:
            checkpoint_path = self.experiments_dir / file_name

            self.logger.info(f'Loading checkpoint "{checkpoint_path}"')

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self._load_state_dict(checkpoint)

            self.logger.info(
                f"""Checkpoint loaded successfully from '{checkpoint_path}' at (epoch {checkpoint['epoch']}) 
                at (iteration {checkpoint['iteration']})\n""")

    def save_checkpoint(self, is_best=False):
        """
        Checkpoint saver
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        checkpoint_path = self.checkpoints_dir / f'epoch_{self.current_epoch}.pth'
        torch.save(self._get_state_dict(), checkpoint_path)
        if is_best:
            best_checkpoint_path = self.checkpoints_dir / 'best.pth'
            torch.save(self._get_state_dict(), best_checkpoint_path)

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            # 'stack' is a dummy context manager
            with ExitStack() as stack:
                if self.debug:
                    # but if in debug mode add an actual manager for anomaly detection
                    stack.enter_context(torch.autograd.detect_anomaly())
                self.train()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")
            self.finalize()

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, self.max_epoch):
            self.train_one_epoch()
            self.validate()
            self.save_checkpoint()

            self.current_epoch += 1

            if self.debug:
                break

    def _log_train_iter(self, **scalars_to_log):
        curr_epoch_iter = self.current_iteration % self.num_train_batches
        self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]{}'.format(
            self.current_epoch,
            curr_epoch_iter * self.dloader_train.batch_size,
            self.num_train_samples,
            100. * curr_epoch_iter / self.num_train_batches,
            ''.join(f'\t{k}: {v:.6f}' for k, v in scalars_to_log.items()),
        ))

        # log to tensorboard
        for scalar_name, scalar_val in scalars_to_log.items():
            self.summary_writer.add_scalar(
                tag=f'train/{scalar_name}',
                scalar_value=scalar_val,
                global_step=self.current_iteration,
            )

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.dloader_train):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.log_interval == 0:
                loss_val = loss.item()
                self._log_train_iter(loss=loss_val)

            self.current_iteration += 1

            if self.debug:
                break

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for data, target in self.dloader_val:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # sum up batch loss
                loss += self.loss_fn(output, target, size_average=False).item()

                if self.debug:
                    break

        loss /= len(self.dloader_val.dataset)

        self.logger.info(f'\nValidation set: Average loss: {loss:.4f}')

        # log to tensorboard
        self.summary_writer.add_scalar(
            tag='validation/loss',
            scalar_value=loss,
            global_step=self.current_epoch,
        )

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass

    @property
    def num_train_samples(self):
        return len(self.dloader_train.dataset)

    @property
    def num_val_samples(self):
        return len(self.dloader_val.dataset)

    @property
    def num_train_batches(self):
        return len(self.dloader_train)
