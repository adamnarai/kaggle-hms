import os
import numpy as np
import pandas as pd
from collections import namedtuple
import wandb

import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold

from dataloader import get_dataloaders, get_datasets
from utils import seed_everything
from trainer import Trainer
from model.model import SpecCNN


def train_model(CFG, data, df_train, df_validation, state_filename, validate=True, wandb_log=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Data loaders
    datasets = get_datasets(CFG, data, df_train, df_validation)
    dataloaders = get_dataloaders(CFG, datasets)

    # Model definition
    model = SpecCNN(model_name=CFG.base_model, num_classes=len(CFG.TARGETS), in_channels=CFG.in_channels, pretrained=CFG.pretrained).to(device)
    
    # Loss function
    if CFG.loss == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)
    elif CFG.loss == 'BCEWithLogitsLoss':
        loss_fn = nn.BCEWithLogitsLoss()
    elif CFG.loss == 'KLDivLoss':
        loss_fn = nn.KLDivLoss(reduction='batchmean')

    # Optimizer
    if CFG.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=CFG.base_lr, momentum=CFG.sgd_momentum)
    elif CFG.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=CFG.base_lr)
    elif CFG.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=CFG.base_lr)
    
    # Scheduler
    if CFG.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=CFG.scheduler_step_size, gamma=CFG.lr_gamma)
    elif CFG.scheduler == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=CFG.base_lr, max_lr=CFG.base_lr*5,
                                                step_size_up=5, cycle_momentum=False, mode='triangular2')
    elif CFG.scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs+CFG.freeze_epochs)
    elif CFG.scheduler == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.base_lr, total_steps=CFG.epochs+CFG.freeze_epochs)

    # Training
    trainer = Trainer(model, dataloaders, loss_fn, optimizer, scheduler, device, state_filename=state_filename, metric='kl_divergence', wandb_log=wandb_log)
    trainer.train_epochs(num_epochs=CFG.epochs, validate=validate)
    trainer.save_state(state_filename)
    return trainer

# Undersample to N trials
def sampler(x, n):
    if len(x) <= n:
        return x
    else:
        return x.sample(n)

def load_data(CFG):
    df = pd.read_csv(os.path.join(CFG.data_dir, 'train.csv'))
    data = dict()
    if CFG.data_type == 'spec' or CFG.data_type == 'spec+eeg_tf':
        data['spec_data'] = np.load(os.path.join(CFG.data_dir, 'spec_data.npy'), allow_pickle=True).item()
    else:
        data['spec_data'] = []
    if CFG.data_type == 'eeg_tf' or CFG.data_type == 'spec+eeg_tf':
        if os.path.isfile(os.path.join(CFG.data_dir, f'{CFG.eeg_tf_data}.npy')):
            data['eeg_tf_data'] = np.load(os.path.join(CFG.data_dir, f'{CFG.eeg_tf_data}.npy'), allow_pickle=True).item()
        elif os.path.isdir(os.path.join(CFG.data_dir, f'{CFG.eeg_tf_data}')):
            data['eeg_tf_data'] = os.path.join(CFG.data_dir, f'{CFG.eeg_tf_data}')
    else:
        data['eeg_tf_data'] = []
    data['eeg_data'] = [] #np.load(os.path.join(data_dir, 'eeg_data.npy'), allow_pickle=True).item()

    # Spectrogram trial selection
    if CFG.spec_trial_selection == 'all':
        pass
    elif CFG.spec_trial_selection == 'first':
        df = df.groupby('spectrogram_id').head(1).reset_index(drop=True)
    elif CFG.spec_trial_selection == 'random':
        df = df.groupby('spectrogram_id').apply(lambda x: sampler(x, CFG.spec_random_trial_num)).reset_index(drop=True)

    # EEG trial selection
    if CFG.eeg_trial_selection == 'all':
        pass
    elif CFG.eeg_trial_selection == 'first':
        df = df.groupby('eeg_id').head(1).reset_index(drop=True)
    elif CFG.eeg_trial_selection == 'random':
        df = df.groupby('eeg_id').apply(lambda x: sampler(x, CFG.eeg_random_trial_num)).reset_index(drop=True)
    elif CFG.eeg_trial_selection == 'first+last':
        df = df.groupby('eeg_id').apply(lambda x: pd.concat((x.head(1), x.tail(1))).reset_index(drop=True))

    # Normalize targets
    df[CFG.TARGETS] /= df[CFG.TARGETS].sum(axis=1).values[:, None]

    return df, data

def init_wandb(CFG):
    wandb.login(key=CFG.wandb_key)
    cfg_dict = dict((key, value) for key, value in dict(CFG.__dict__).items() if not callable(value) and not key.startswith('__'))
    run = wandb.init(project=f'kaggle-{CFG.project_name}', config=cfg_dict, tags=CFG.tags, notes=CFG.notes)
    return run

def create_cv_splits(CFG, df):
    skf = StratifiedGroupKFold(n_splits=CFG.cv_fold, random_state=CFG.seed, shuffle=True)
    splits = []
    for train_index, valid_index in skf.split(X=np.zeros(len(df)), y=df[CFG.stratif_vars], groups=df[CFG.grouping_vars]):
        splits.append((df.iloc[train_index].copy(), df.iloc[valid_index].copy()))
    return splits

def save_splits(CFG, splits, model_name):
    df = pd.DataFrame()
    for i, (df_train, df_validation) in enumerate(splits):
        df_train.loc[:,'split'] = 'train'
        df_train.loc[:,'fold'] = i + 1
        df_validation.loc[:,'split'] = 'validation'
        df_validation.loc[:,'fold'] = i + 1
        df = pd.concat([df, df_train, df_validation])
    df = df.sort_values(by=['fold', 'split']).reset_index(drop=True)
    df.to_csv(os.path.join(CFG.results_dir, 'models', model_name, 'splits.csv'), index=False)

def train(CFG):
    # Wandb
    if CFG.use_wandb:
        run = init_wandb(CFG)
    else:
        WandbRun = namedtuple('WandbRun', 'name')
        run = WandbRun('debug')

    # Create model dir
    model_name = f'{CFG.project_name}-{run.name}'
    os.makedirs(os.path.join(CFG.results_dir, 'models', model_name), exist_ok=True)

    # Seed
    seed_everything(CFG.seed)

    # Load data
    df, data = load_data(CFG)

    # Cross-validation splicts
    splits = create_cv_splits(CFG, df)
    save_splits(CFG, splits, model_name)

    metric_list = []
    best_metric_list = []
    for cv, (df_train, df_validation) in enumerate(splits):
        print(f"Cross-validation fold {cv+1}/{CFG.cv_fold}")
        state_filename = os.path.join(CFG.results_dir, 'models', model_name, f'{model_name}-cv{cv+1}.pt')
        if CFG.use_wandb and cv == 0:
            wandb_log = True
        else:
            wandb_log = False
        trainer = train_model(CFG, data, df_train, df_validation, state_filename, wandb_log=wandb_log)
        best_metric_list.append(trainer.best_metric)
        metric_list.append(trainer.metric)
        if CFG.use_wandb:
            wandb.log({f'metric_cv{cv+1}': trainer.metric})
            wandb.log({f'best_metric_cv{cv+1}': trainer.best_metric})
        if CFG.one_fold:
            break
    if CFG.use_wandb:
        wandb.log({f'mean_metric': np.mean(metric_list)})
        wandb.log({f'mean_best_metric': np.mean(best_metric_list)})
        wandb.finish()

    # Final training on all data
    if CFG.train_full_model:
        state_filename = os.path.join(CFG.results_dir, 'models', f'{model_name}-full.pt')
        trainer = train_model(CFG, data, df, df, state_filename, validate=False, wandb_log=False)
        if CFG.use_wandb:
            wandb.finish()