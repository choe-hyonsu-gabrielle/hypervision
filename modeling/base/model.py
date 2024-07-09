import abc
from typing import Any
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from transformers import PreTrainedModel, PreTrainedTokenizer
from modeling.base.config import ModelConfigBase


class LightningModuleBase(pl.LightningModule):
    # This is a base class of customized LightningModule variants with common features are pre-implemented.
    def __init__(self, config: ModelConfigBase, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # model configs & peripherals
        self.config = config

        # core components
        self.pretrained_tokenizer: PreTrainedTokenizer = self.config.pretrained_tokenizer
        self.pretrained_model: PreTrainedModel = self.config.pretrained_model.to(self.device)

    def configure_optimizers(self):
        optimizer = self.config.optimizer(self.parameters(), lr=self.config.learning_rate)
        if self.config.lr_scheduler:
            lr_scheduler_config = dict(
                scheduler=self.config.lr_scheduler(optimizer, **self.config.lr_scheduler_params),
                interval='step',
                name=self.config.lr_scheduler.__class__.__name__
            )
            return [optimizer], [lr_scheduler_config]
        return optimizer

    @abc.abstractmethod
    def batch_forward_and_loss(self, samples: Any) -> tuple[torch.Tensor, torch.Tensor]:
        # make it returns a tuple of loss and logit like,
        # return loss, logit
        pass

    def training_step(self, samples, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        loss, _ = self.batch_forward_and_loss(samples)
        self.log(name='train/loss', value=loss, prog_bar=True, sync_dist=True)
        if self.config.lr_scheduler:
            self.log('learning_rate', self.lr_schedulers().get_last_lr()[0], prog_bar=True, sync_dist=True)
        return dict(loss=loss)

    def validation_step(self, samples, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        loss, _ = self.batch_forward_and_loss(samples)
        self.log(name='valid/loss', value=loss, batch_size=self.config.batch_size, prog_bar=True, sync_dist=True)
        return dict(loss=loss)

    def test_step(self, samples, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        loss, _ = self.batch_forward_and_loss(samples)
        self.log(name='test/loss', value=loss, batch_size=self.config.batch_size, prog_bar=True, sync_dist=True)
        return dict(loss=loss)
