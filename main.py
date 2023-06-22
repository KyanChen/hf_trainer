#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Dict, Sequence

import torch
import transformers
from mmengine import Config, MODELS, DATASETS, FUNCTIONS
from torch.utils.data import Dataset
from transformers import Trainer
import vqvae

@dataclass
class ModelArguments:
    block_size = 64
    nb_joints = 21
    model_cfg = dict(
        type='MotionVQVQEPLer',
        data_preprocessor=dict(
            type='NormalizationMotion',
            mean_std_file=f'data/motion/kit_train_mean_std_info_{block_size}.pkl',
        ),
        backbone=dict(
            type='HumanVQVAE',
            quantizer='ema_reset',
            in_channel=251,  # 263
            nb_code=512,
            code_dim=512,
            output_emb_width=512,
            down_t=2,
            stride_t=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            activation='relu',
            norm=None
        ),
        head=dict(
            type='MotionVQVAEPseudoHead',
            nb_joints=nb_joints,
            commit_loss_weight=0.02,
            losses=dict(
                motion_loss=dict(type='SmoothL1Loss', loss_weight=1.0),
                motion_vec_loss=dict(type='SmoothL1Loss', loss_weight=0.1)
            ),
        )
    )
    model_cfg = Config(model_cfg)


@dataclass
class DataArguments:
    data_root = 'data'
    block_size = 64
    train_dataset_cfg = dict(
        type='VQMotionDataset',
        data_root=data_root,
        ann_file='cky_trainval.txt',
        # dataset_name='kit',
        block_size=block_size,
        n_offset=1,
        test_mode=False,
    )
    train_dataset_cfg = Config(train_dataset_cfg)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns = False
    optim: str = field(default="adamw_torch")
    train_batch_size = 4


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def make_supervised_data_module(data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = DATASETS.build(data_args.train_dataset_cfg)

    collate_fn_cfg = dict(type='default_collate')
    collate_fn_type = collate_fn_cfg.pop('type')
    collate_fn = FUNCTIONS.get(collate_fn_type)
    data_collator = partial(collate_fn, **collate_fn_cfg)

    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False

    model = MODELS.build(model_args.model_cfg)

    data_module = make_supervised_data_module(data_args=data_args)
    trainer = Trainer(model=model, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
