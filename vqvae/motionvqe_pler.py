from typing import Any
import mmengine
from mmengine.model import BaseModel
import torch
import torch.nn as nn
from mmengine.registry import MODELS
from transformers import PreTrainedModel


@MODELS.register_module()
class MotionVQVQEPLer(BaseModel):
    def __init__(self,
                 data_preprocessor,
                 backbone,
                 head,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_preprocessor = MODELS.build(data_preprocessor)
        self.backbone = MODELS.build(backbone)
        self.head = MODELS.build(head)

    def training_val_step(self, batch, batch_idx=0, prefix=''):
        if hasattr(self, 'data_preprocessor'):
            data = self.data_preprocessor(batch)
        gt_motion = data['inputs']
        pred_motion, loss_commit, perplexity = self.backbone(gt_motion)

        losses = self.head.loss(
            pred_motion=pred_motion,
            loss_commit=loss_commit,
            perplexity=perplexity,
            gt_motion=gt_motion,
        )

        parsed_losses, log_vars = self.parse_losses(losses)
        log_vars = {f'{prefix}_{k}': v for k, v in log_vars.items()}
        log_vars['loss'] = parsed_losses

        # self.log_dict(log_vars, prog_bar=True)
        return log_vars

    def validation_step(self, batch, batch_idx=0):
        return self.training_val_step(batch, batch_idx, prefix='val')

    def training_step(self, batch, batch_idx=0):
        return self.training_val_step(batch, batch_idx, prefix='train')

    def forward(self, **batch):
        return self.training_val_step(batch, prefix='train')

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if hasattr(self, 'data_preprocessor'):
            data = self.data_preprocessor(batch)
        gt_motion = data['inputs']

        pred_motion, loss_commit, perplexity = self.backbone(gt_motion)
        pred_denorm = self.data_preprocessor.denormalize(pred_motion)
        return pred_denorm

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if hasattr(self, 'data_preprocessor'):
            data = self.data_preprocessor(batch)
        gt_motion = data['inputs']

        pred_tokens = self.backbone.encode(gt_motion)
        return pred_tokens





