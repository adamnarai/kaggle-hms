import os
from PIL import Image
import numpy as np
import glob
import random
from skimage.transform import resize

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SpecDataset(Dataset):
    """HMS spectrogram dataset dataset."""

    def __init__(self, CFG, df, data, transform=None):
        self.df = df
        self.spec_ids = self.df['spectrogram_id'].values
        self.eeg_ids = self.df['eeg_id'].values
        self.targets = self.df[CFG.TARGETS].values
        self.spec_offsets = (self.df['spectrogram_label_offset_seconds'].values // 2).astype(int)
        self.eeg_offsets = self.df['eeg_label_offset_seconds'].values
        self.spec_data = data['spec_data']
        self.eeg_data = data['eeg_data']
        self.eeg_tf_data = data['eeg_tf_data']
        self.transform = transform
        self.data_type = CFG.data_type # 'spec' or 'eeg_tf'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load spec image
        spec_id = self.spec_ids[idx]
        offset = self.spec_offsets[idx]
        spec_image = self.spec_data[spec_id]

        spec_image = spec_image[offset:offset+300, :].T

        # log transform spectogram
        spec_image = np.clip(spec_image, np.exp(-4), np.exp(8))
        spec_image = np.log(spec_image)

        # standardize per image
        ep = 1e-6
        m = np.nanmean(spec_image.flatten())
        s = np.nanstd(spec_image.flatten())
        spec_image = (spec_image-m)/(s+ep)
        spec_image = np.nan_to_num(spec_image, nan=0.0)

        spec_image = spec_image[..., None]


        # Load eeg tf image
        eeg_id = self.eeg_ids[idx]
        eeg_offset = self.eeg_offsets[idx]
        eeg_tf_image = self.eeg_tf_data[eeg_id]

        # # log transform spectogram
        # eeg_tf_image = np.clip(eeg_tf_image, np.exp(-4), np.exp(8))
        # eeg_tf_image = np.log(eeg_tf_image)

        # # standardize per image
        # ep = 1e-6
        # m = np.nanmean(eeg_tf_image.flatten())
        # s = np.nanstd(eeg_tf_image.flatten())
        # eeg_tf_image = (eeg_tf_image-m)/(s+ep)
        # eeg_tf_image = np.nan_to_num(eeg_tf_image, nan=0.0)

        eeg_tf_image = eeg_tf_image[..., None]

        # Final image
        if self.data_type == 'spec':
            image = spec_image
        elif self.data_type == 'eeg_tf':
            image = eeg_tf_image
        elif self.data_type == 'spec+eeg_tf':
            image = np.concatenate([resize(spec_image, (512, 512)), resize(eeg_tf_image, (512, 512))], axis=2)

        # image = np.concatenate([image, image, image], axis=2)
        # image = np.concatenate((image[:100, :], image[100:200, :], image[200:300, :], image[300:400, :]), axis=2)
        
        if self.transform:
            image = self.transform(image=image)['image']

        return image, torch.from_numpy(self.targets[idx])

def get_datasets(CFG, data, df_train, df_validation):
    transform = {
    'train':
    A.Compose([
        A.CoarseDropout(**CFG.coarse_dropout_args),
        A.ColorJitter(**CFG.color_jitter_args),
        ToTensorV2(),
        # transforms.Normalize(mean=CFG['img_color_mean'], std=CFG['img_color_std']),
        # transforms.RandomErasing(p=CFG['random_erasing_p'])
    ]),
    'validation':
     A.Compose([
        # A.Resize(height=CFG['img_size'], width=CFG['img_size']),
        ToTensorV2(),
        # transforms.Normalize(mean=CFG['img_color_mean'], std=CFG['img_color_std'])
    ])}
    
    train_dataset = SpecDataset(CFG, 
                                df=df_train, 
                                data=data,
                                transform=transform['train'])
    validation_dataset = SpecDataset(CFG, 
                                     df=df_validation, 
                                     data=data,
                                     transform=transform['validation'])
    datasets = {'train': train_dataset, 'validation': validation_dataset}
    return datasets

def get_dataloaders(CFG, datasets):
    train_loader = DataLoader(datasets['train'], batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.dataloader_num_workers, pin_memory=True)
    validation_loader = DataLoader(datasets['validation'], batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.dataloader_num_workers, pin_memory=True)
    dataloaders = {'train': train_loader, 'validation': validation_loader}
    return dataloaders