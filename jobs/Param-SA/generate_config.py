import numpy as np
import os
import yaml

beta_ood_list = [1, 0.1, 0.01, 0.001]
beta_z_list = [1, 0.1, 0.01, 0.001]

config_dir = os.path.join('config', 'GAN', 'SA')

for beta_ood in beta_ood_list:
    for beta_z in beta_z_list:
        path = os.path.join(config_dir, f'OOD-GAN-FashionMNIST-{beta_ood}-{beta_z}.yaml')
        config = {
            "dataset": {
                "dset": "FashionMNIST",
                "is_within_dset": True,
                "ind": [0, 1, 2, 3, 4, 5, 6, 7],
                "ood": [8, 9]
            },
            "path": {
                "root_dir": "checkpoint/",
                "pretrained_dir": "checkpoint/pretrained/",
                "sample_dir": "checkpoint/OOD-Sample/"
            },
            "experiment": {
                "method": "OOD-GAN",
                "regime": "Balanced",
                "observed_cls": None
            },
            "dset_info": {
                "img_info": {
                    "H": 28,
                    "W": 28,
                    "C": 1
                },
                "num_class": 8
            },
            "model": {
                "D": "DenseNet",
                "D_config": {
                    "depth": 100
                },
                "G": "Deep_G",
                "G_config": {
                    "noise_dim": 96
                }
            },
            "train_config": {
                "score": "Wasserstein",
                "mc": 1,
                "T": 1,
                "max_epochs": 16,
                "bsz_tri": 64,
                "bsz_val": 64,
                "ood_bsz": 32,
                "optimizer": {
                    "d_lr": 0.001,
                    "g_lr": 0.001,
                    "beta1": 0.5,
                    "beta2": 0.999
                },
                "hp": {
                    "ce": 1,
                    "wass": beta_ood,
                    "dist": 1
                },
                "scaling": beta_z / beta_ood,
                "d_step_ratio": 1,
                "g_step_ratio": 1,
                "noise_dim": 96,
                "logging": {
                    "n_steps_log": 20,
                    "n_epochs_save": 4
                }
            },
            "eval_config": {
                "each_cls": False,
                "cls_idx": [8, 9],
                "n_lr": 5000
            }
        }

        with open(path, "w") as f:
            yaml.dump(config, f, sort_keys=False)
