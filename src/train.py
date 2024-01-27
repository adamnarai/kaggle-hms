import os
import numpy as np
import pandas as pd
from collections import namedtuple
import wandb

import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold

from dataloader import get_spec_dataloaders, get_spec_datasets
from utils import seed_everything
from trainer import Trainer
from model.model import SpecCNN

TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

# Paths
root = '/media/latlab/MR/projects/kaggle-hms'
data_dir = os.path.join(root, 'data')
results_dir = os.path.join(root, 'results')
train_eeg_dir = os.path.join(data_dir, 'train_eegs')
train_spectrogram_dir = os.path.join(data_dir, 'train_spectrograms')


def train_model(CFG, spec_data, df_train, df_validation, state_filename, validate=True, wandb_log=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Data loaders
    datasets = get_spec_datasets(CFG, spec_data, df_train, df_validation)
    dataloaders = get_spec_dataloaders(CFG, datasets)

    # Model definition
    model = SpecCNN(model_name=CFG['base_model'], num_classes=len(TARGETS), in_channels=CFG['in_channels']).to(device)
    
    # Loss function
    if CFG['loss'] == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss(label_smoothing=CFG['label_smoothing'])
    elif CFG['loss'] == 'BCEWithLogitsLoss':
        loss_fn = nn.BCEWithLogitsLoss()
    elif CFG['loss'] == 'KLDivLoss':
        loss_fn = nn.KLDivLoss(reduction='batchmean')

    # Optimizer
    if CFG['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=CFG['base_lr'], momentum=CFG['sgd_momentum'])
    elif CFG['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=CFG['base_lr'])
    elif CFG['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=CFG['base_lr'])
    
    # Scheduler
    if CFG['scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=CFG['scheduler_step_size'], gamma=CFG['lr_gamma'], verbose=True)
    elif CFG['scheduler'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=CFG['base_lr'], max_lr=CFG['base_lr']*5,
                                                step_size_up=5, cycle_momentum=False, mode='triangular2', verbose=True)
    elif CFG['scheduler'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs']+CFG['freeze_epochs'], verbose=True)
    elif CFG['scheduler'] == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG['base_lr'], total_steps=CFG['epochs']+CFG['freeze_epochs'], verbose=True)

    # Training
    trainer = Trainer(model, dataloaders, loss_fn, optimizer, scheduler, device, state_filename=state_filename, metric='kl_divergence', wandb_log=wandb_log)
    trainer.train_epochs(num_epochs=CFG['epochs'], validate=validate)
    trainer.save_state(state_filename)
    return trainer

def train(CFG, tags, notes, train_final_model=False, use_wandb=True, one_fold=False):
    # Wandb
    if use_wandb:
        wandb.login(key='1b0401db7513303bdea77fb070097f9d2850cf3b')
        run = wandb.init(project='kaggle-hms', config=CFG, tags=tags, notes=notes)
    else:
        WandbRun = namedtuple('WandbRun', 'name')
        run = WandbRun('debug')

    # Seed
    seed_everything(CFG['seed'])

    # Load data
    df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    spec_data = np.load(os.path.join(data_dir, 'spec_data.npy'), allow_pickle=True).item()

    df[TARGETS] /= df[TARGETS].sum(axis=1).values[:, None]
    df = df.groupby("spectrogram_id").head(1).reset_index(drop=True)

    skf = StratifiedGroupKFold(n_splits=CFG['cv_fold'], random_state=CFG['seed'], shuffle=True)
    metric_list = []
    for cv, (train_index, valid_index) in enumerate(skf.split(X=np.zeros(len(df['expert_consensus'])), y=df['expert_consensus'], groups=df['patient_id'])):
        print(f"Cross-validation fold {cv+1}/{CFG['cv_fold']}")
        df_train = df.iloc[train_index]
        df_validation = df.iloc[valid_index]
        run_name = f'{run.name}-cv{cv+1}'
        state_filename = os.path.join(results_dir, 'models', f'ubc-ocean-{run_name}.pt')
        if use_wandb and cv == 0:
            wandb_log = True
        else:
            wandb_log = False
        trainer = train_model(CFG, spec_data, df_train, df_validation, state_filename, wandb_log=wandb_log)
        metric_list.append(trainer.best_metric)
        if use_wandb:
            wandb.log({f'kl_div_cv{cv+1}': trainer.best_metric})
        if one_fold:
            break
    if use_wandb:
        wandb.log({f'mean_kl_div': np.mean(metric_list)})
        wandb.finish()

    # Final training on all data
    if train_final_model:
        state_filename = os.path.join(results_dir, 'models', f'ubc-ocean-{run.name}.pt')
        trainer = train_model(CFG, spec_data, df, df, state_filename, validate=False, wandb_log=False)
        if use_wandb:
            wandb.finish()