import shutil
import yaml
import os
import numpy as np
from pathlib import Path
import itertools

# preprocess_model
from preprocess_model.image_preprocess_model import DIVAESR
from preprocess_model.preprocess_data import DIVAESRDataLoader
from preprocess_model.preprocess_experiment import Prexpriment

# SR model
from preprocess_model.configs.option import args
import SR_model.utility as utility

# pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
import torch.nn as nn


def save_config(config, save_path):
    with open(save_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


if __name__ == '__main__':
    sr_model_args = args
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    vae_model_args = config['model_params']

    # fix the seed for reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)
    # 网格搜索配置
    train_batch_sizes = config['data_params']['train_batch_size']
    learning_rates = config['exp_params']['LR']
    max_epochs_list = config['trainer_params']['max_epochs']
    # 创建参数组合
    param_combinations = list(itertools.product(train_batch_sizes, learning_rates, max_epochs_list))
    # 定义保存配置文件的目录
    configs_save_dir = "saved_configs"
    os.makedirs(configs_save_dir, exist_ok=True)

    for idx, (train_batch_size, learning_rate, max_epochs) in enumerate(param_combinations):
        # 更新配置
        config['data_params']['train_batch_size'] = train_batch_size
        config['exp_params']['LR'] = learning_rate
        config['trainer_params']['max_epochs'] = max_epochs
        # 生成新的配置文件名称
        config_filename = f"config_{idx}.yaml"
        config_save_path = os.path.join(configs_save_dir, config_filename)
        # 保存新的配置文件
        save_config(config, config_save_path)
        # define tb_logger
        tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'], name=config['model_params']['name'], )
        # define model
        preprocess_model = DIVAESR(sr_model_args, vae_model_args)
        # 打印整个模型的结构
        print(preprocess_model)
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for key in state_dict:
                new_key = key.replace("model.", "")  # Remove the "model." prefix
                new_state_dict[new_key] = state_dict[key]
            preprocess_model.load_state_dict(new_state_dict)
        
        # 计算并打印模型的参数量
        total_params = sum(p.numel() for p in preprocess_model.parameters())
        print(f"Total Parameters: {total_params}")
        experiment = Prexpriment(preprocess_model, config['exp_params'])
        
        # todo: define dataset and dataloader
        data = DIVAESRDataLoader(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
        data.setup()
        
        # todo: define trainval one epoch
        runner = Trainer(logger=tb_logger,
                         callbacks=[
                         LearningRateMonitor(),
                         ModelCheckpoint(save_top_k=2,
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                         monitor="val_loss",
                                         save_last=True),
                     ],
                     # strategy=DDPPlugin(find_unused_parameters=False),
                     **config['trainer_params'])
        
        Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
        Path(f"{tb_logger.log_dir}/VAE_Reconstructions").mkdir(exist_ok=True, parents=True)
        Path(f"{tb_logger.log_dir}/SR_Reconstructions").mkdir(exist_ok=True, parents=True)
        
        print(f"======= Training {config['model_params']['name']} =======")
        runner.fit(experiment, datamodule=data)
        # 保存模型和日志文件
        shutil.copy(config_save_path, tb_logger.log_dir)
        shutil.copy('preprocess_model/configs/option.py', tb_logger.log_dir)
