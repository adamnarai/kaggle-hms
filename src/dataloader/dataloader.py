import os
import numpy as np
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import torchvision.transforms.functional as F

class HMSDataset(Dataset):
    """HMS spectrogram dataset dataset."""

    def __init__(self, CFG, df, data, transform=None):
        self.df = df
        self.spec_ids = self.df['spectrogram_id'].values
        self.eeg_ids = self.df['eeg_id'].values
        self.eeg_sub_ids = self.df['eeg_sub_id'].values
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
        if self.data_type == 'spec' or self.data_type == 'spec+eeg_tf':
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
        if self.data_type == 'eeg_tf' or self.data_type == 'spec+eeg_tf':
            eeg_id = self.eeg_ids[idx]
            eeg_sub_id = self.eeg_sub_ids[idx]
            if isinstance(self.eeg_tf_data, dict):
                eeg_tf_image = self.eeg_tf_data[eeg_id]
            elif os.path.isdir(self.eeg_tf_data):
                eeg_tf_image = np.load(os.path.join(self.eeg_tf_data, f'{eeg_id}_{eeg_sub_id}.npy'), allow_pickle=True)

            eeg_tf_image = eeg_tf_image[..., None]

            # # Make it 3 channel
            # eeg_tf_image = np.repeat(eeg_tf_image, 3, axis =-1)


        # Combination of spec and eeg_tf
        if self.data_type == 'spec+eeg_tf':
            if self.transform:
                spec_image = self.transform(spec_image)
                eeg_tf_image = self.transform(eeg_tf_image)
            return spec_image, eeg_tf_image, torch.from_numpy(self.targets[idx])

        # Final image
        if self.data_type == 'spec':
            image = spec_image
        elif self.data_type == 'eeg_tf':
            image = eeg_tf_image
        
        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(self.targets[idx])
    
class RandomChErease(object):
    def __init__(self, eeg_ch_num, drop_ch_num=1, fill_value=0, p=0.5):
        self.eeg_ch_num = eeg_ch_num
        self.drop_ch_num = drop_ch_num
        self.fill_value = fill_value
        self.p = p

    def __call__(self, image):
        if self.p == 0.0 or random.random() >= self.p:
            return image
        
        ch_size = image.shape[1] // self.eeg_ch_num
        ch_idx_list = random.sample(range(self.eeg_ch_num), self.drop_ch_num)
        for ch_idx in ch_idx_list:
            image[:, (ch_idx*ch_size):((ch_idx+1)*ch_size), :] = self.fill_value
        return image
    
class RandomTimeMasking(object):
    def __init__(self, width_prop, erase_num=1, fill_value=0, p=0.5):
        self.width_prop = width_prop
        self.erase_num = erase_num
        self.fill_value = fill_value
        self.p = p

    def __call__(self, image):
        if self.p == 0.0 or random.random() >= self.p:
            return image
        
        width = int(image.shape[2] * self.width_prop)
        for _ in range(self.erase_num):
            time_start = random.randint(0, image.shape[2] - width)
            image[..., time_start:(time_start+width)] = self.fill_value
        return image
    
class RandomFrequencyMasking(object):
    def __init__(self, bandwidth_prop, eeg_ch_num, erase_num=1, fill_value=0, p=0.5):
        self.bandwidth_prop = bandwidth_prop
        self.eeg_ch_num = eeg_ch_num
        self.erase_num = erase_num
        self.fill_value = fill_value
        self.p = p

    def __call__(self, image):
        if self.p == 0.0 or random.random() >= self.p:
            return image
        
        ch_size = image.shape[1] // self.eeg_ch_num
        bandwidth = int(ch_size * self.bandwidth_prop)
        for _ in range(self.erase_num):
            freq_start = random.randint(0, ch_size - bandwidth)
            for ch_idx in range(self.eeg_ch_num):
                image[:, (ch_idx*ch_size+freq_start):(ch_idx*ch_size+freq_start+bandwidth), :] = self.fill_value
        return image

def get_datasets(CFG, data, df_train, df_validation):
    transform = {
    'train':
    v2.Compose([
        v2.ToImage(),
        # v2.Resize(height=56, width=500),
        # v2.RandomApply(nn.ModuleList([v2.GaussianBlur(**CFG.gaussian_blur_args)]), p=CFG.gaussian_blur_p),
        RandomChErease(**CFG.random_ch_erease_args),
        RandomTimeMasking(**CFG.random_time_masking_args),
        RandomFrequencyMasking(**CFG.random_frequency_masking_args),
        # v2.RandomErasing(**CFG.random_erasing_args),
        v2.ToDtype(torch.float32, scale=True),
    ]),
    'validation':
     v2.Compose([
        v2.ToImage(),
        # v2.Resize(height=512, width=512),
        v2.ToDtype(torch.float32, scale=True)
    ])}
    
    train_dataset = HMSDataset(CFG, 
                                df=df_train, 
                                data=data,
                                transform=transform['train'])
    validation_dataset = HMSDataset(CFG, 
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