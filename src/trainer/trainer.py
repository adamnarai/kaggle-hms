import os
import time
from tempfile import TemporaryDirectory
import warnings
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np
import wandb
from tqdm import tqdm
import pandas as pd
from ext.kaggle_kl_div.kaggle_kl_div import score as kaggle_kl_div_score

import torch
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, dataloaders, loss_fn, optimizer, scheduler, device, state_filename, metric, num_epochs=10, wandb_log=False, loss_fn_val=None):
        self.model = model
        self.train_dataloader = dataloaders['train']
        self.test_dataloader = dataloaders['validation']
        self.loss_fn = loss_fn
        self.loss_fn_val = loss_fn_val
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.state_filename = state_filename
        self.metric = metric
        self.num_epochs = num_epochs
        self.wandb_log = wandb_log
        self.best_metric = np.inf
        self.last_metric = np.inf
        self.epoch_count = 0
        self.test_y = []
        self.test_pred = []

    def train_epochs(self, num_epochs=None, validate=True):
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if num_epochs == 0:
            return self.model, 0
        since = time.time()

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            print('-' * 10)

            train_loss = self.train()
            if validate:
                test_loss, metric = self.test()
                self.last_metric = metric
                if metric < self.best_metric:
                    self.best_metric = metric
                    print(f"New best {self.metric}: {self.best_metric:.4f}\nSaving model to {self.state_filename}")
                    self.save_state(self.state_filename.replace('.pt', '_best.pt'))
            else:
                test_loss, metric = np.nan, np.nan
                self.best_metric = np.nan
            self.scheduler.step()
            print(f"lr: {self.scheduler.get_last_lr()}")
            print(f"train loss: {train_loss:.4f}, test loss: {test_loss:.4f}, {self.metric}: {metric:.4f}\n")
            if self.wandb_log:
                wandb.log({'train_loss': train_loss, 'valid_loss': test_loss, self.metric: metric})

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Final {self.metric}: {metric:4f}\n')
        self.epoch_count += self.num_epochs
        return
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
        return
    
    def train(self):
        num_batches = len(self.train_dataloader)
        self.model.train()
        
        train_loss = 0
        for batch in tqdm(self.train_dataloader, total=num_batches):
            if len(batch) == 2:
                X, y = batch
                X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                pred = self.model(X)
            elif len(batch) == 3:
                X1, X2, y = batch
                X1, X2, y = X1.to(self.device, non_blocking=True), X2.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                pred = self.model(X1, X2)

            # Compute prediction error
            loss = self.loss_fn(F.log_softmax(pred, dim=-1), y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()

        train_loss /= num_batches
        return train_loss

    def test(self):
        num_batches = len(self.test_dataloader)
        self.model.eval()
        
        test_loss = 0
        metric = 0
        test_y = []
        test_pred = []
        cumm_pred = []
        cumm_y = []
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, total=num_batches):
                if len(batch) == 2:
                    X, y = batch
                    X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                    pred = self.model(X)
                elif len(batch) == 3:
                    X1, X2, y = batch
                    X1, X2, y = X1.to(self.device, non_blocking=True), X2.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                    pred = self.model(X1, X2)

                test_loss += self.loss_fn(F.log_softmax(pred, dim=-1), y).item()
                if self.metric == 'balanced_accuracy':
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=UserWarning)
                        metric += balanced_accuracy_score(y.cpu().numpy(), pred.cpu().numpy())
                elif self.metric == 'accuracy':
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=UserWarning)
                        metric += accuracy_score(y.cpu().numpy(), pred.cpu().numpy())
                elif self.metric == 'kl_divergence':
                    cumm_pred.append(F.softmax(pred, dim=-1).cpu().numpy())
                    cumm_y.append(y.cpu().numpy())
                else:
                    raise NotImplementedError(f"Metric {self.metric} not implemented")

        test_loss /= num_batches

        if self.metric == 'kl_divergence':
            cumm_pred = np.concatenate(cumm_pred)
            cumm_pred_df = pd.DataFrame(cumm_pred)
            cumm_pred_df['id'] = np.arange(len(cumm_pred_df))
            cumm_y = np.concatenate(cumm_y)
            cumm_y_df = pd.DataFrame(cumm_y)
            cumm_y_df['id'] = np.arange(len(cumm_y_df))
            metric = kaggle_kl_div_score(submission=cumm_pred_df, solution=cumm_y_df, row_id_column_name='id')
        else:
            metric /= num_batches

        return test_loss, metric
    
    def save_state(self, filename):
        torch.save(self.model.state_dict(), filename)
        return
    
    def load_state(self, filename):
        self.model.load_state_dict(torch.load(filename))
    
    def load_best_state(self):
        self.load_state(self.state_filename.replace('.pt', '_best.pt'))
    
    def get_model(self):
        return self.model
    
    def set_dataloaders(self, dataloaders):
        self.train_dataloader = dataloaders['train']
        self.test_dataloader = dataloaders['validation']
