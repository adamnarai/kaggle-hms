import os
from PIL import Image
import numpy as np
import glob
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SpecDataset(Dataset):
    """HMS spectrogram dataset dataset."""

    TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

    def __init__(self, df, spec_data, transform=None):
        self.df = df
        self.spec_ids = self.df['spectrogram_id'].values
        self.targets = self.df[self.TARGETS].values
        self.spec_offsets = (self.df['spectrogram_label_offset_seconds'].values // 2).astype(int)
        self.spec_data = spec_data
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        spec_id = self.spec_ids[idx]
        offset = self.spec_offsets[idx]
        image = self.spec_data[spec_id]

        image = image[offset:offset+300, :].T

        # log transform spectogram
        image = np.clip(image, np.exp(-4), np.exp(8))
        image = np.log(image)

        # standardize per image
        ep = 1e-6
        m = np.nanmean(image.flatten())
        s = np.nanstd(image.flatten())
        image = (image-m)/(s+ep)
        image = np.nan_to_num(image, nan=0.0)
        
        if self.transform:
            image = self.transform(image)

        # Create 3-channel monochrome image
        image = torch.cat([image, image, image], dim=0)

        return image, torch.from_numpy(self.targets[idx])

def get_spec_datasets(CFG, spec_data, df_train, df_validation):
    transform = {
    'train':
    transforms.Compose([
        # transforms.Resize((CFG['img_size'], CFG['img_size'])),
        # transforms.ColorJitter(**CFG['color_jitter']),
        transforms.ToTensor(),
        # transforms.Normalize(mean=CFG['img_color_mean'], std=CFG['img_color_std']),
        # transforms.RandomErasing(p=CFG['random_erasing_p'])
    ]),
    'validation':
     transforms.Compose([
        # transforms.Resize((CFG['img_size'], CFG['img_size'])),
        transforms.ToTensor(),
        # transforms.Normalize(mean=CFG['img_color_mean'], std=CFG['img_color_std'])
    ])}
    
    train_dataset = SpecDataset(df=df_train, 
                        spec_data=spec_data,
                        transform=transform['train'])
    validation_dataset = SpecDataset(df=df_validation, 
                        spec_data=spec_data,
                        transform=transform['validation'])
    datasets = {'train': train_dataset, 'validation': validation_dataset}
    return datasets

def get_spec_dataloaders(CFG, datasets):
    train_loader = DataLoader(datasets['train'], batch_size=CFG['batch_size'], shuffle=True, num_workers=CFG['dataloader_num_workers'], pin_memory=True)
    validation_loader = DataLoader(datasets['validation'], batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['dataloader_num_workers'], pin_memory=True)
    dataloaders = {'train': train_loader, 'validation': validation_loader}
    return dataloaders