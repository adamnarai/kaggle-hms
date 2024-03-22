import os
import numpy as np
import random
from functools import partial
import mne

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import torchvision.transforms.functional as F
from torch.utils.data import default_collate
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
        self.data_type = CFG.data_type
        self.ch_list = CFG.ch_list
        self.ch_pairs = CFG.ch_pairs
        self.filt_hp = CFG.filt_hp
        self.drop_ecg = CFG.drop_ecg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load spec image
        if self.data_type == 'spec' or self.data_type == 'spec+eeg_tf' or self.data_type == 'spec+eeg_tf+eeg':
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
        if self.data_type == 'eeg_tf' or self.data_type == 'spec+eeg_tf' or self.data_type == 'spec+eeg_tf+eeg':
            eeg_id = self.eeg_ids[idx]
            if isinstance(self.eeg_tf_data, dict):
                eeg_tf_image = self.eeg_tf_data[eeg_id]
            elif os.path.isdir(self.eeg_tf_data):
                eeg_sub_id = self.eeg_sub_ids[idx]
                eeg_tf_image = np.load(os.path.join(self.eeg_tf_data, f'{eeg_id}_{eeg_sub_id}.npy'), allow_pickle=True)

            eeg_tf_image = eeg_tf_image[..., None]

            if self.drop_ecg:
                eeg_tf_image = eeg_tf_image[:(18*28), ...]
        
        # Load raw eeg data
        if self.data_type == 'eeg' or self.data_type == 'spec+eeg_tf+eeg':
            mne.set_log_level('warning')
            eeg_id = self.eeg_ids[idx]
            eeg_raw = self.eeg_data[eeg_id]
            eeg_raw = np.nan_to_num(eeg_raw, nan=0.0)

            # Dipoles
            arr_list = []
            for (ch1, ch2) in self.ch_pairs:
                arr_list.append(eeg_raw[:,self.ch_list.index(ch1)] - eeg_raw[:,self.ch_list.index(ch2)])
            # arr_list.append(eeg_raw[:,self.ch_list.index('EKG')])
            eeg_raw = np.stack(arr_list, axis=0)
            ch_names = ['-'.join(pair) for pair in self.ch_pairs] #+ ['EKG']

            # LP filter, crop, downsample
            raw = mne.io.RawArray(eeg_raw, mne.create_info(ch_names=ch_names, sfreq=200, ch_types=['eeg']*(len(ch_names)-1) + ['ecg']))
            raw = raw.filter(self.filt_hp, None, picks=['eeg', 'ecg'])
            raw = raw.crop(tmin=self.eeg_offsets[idx], tmax=self.eeg_offsets[idx] + 50, include_tmax=False)
            # raw = raw.resample(100)
            eeg_raw = raw.get_data()

            # Clip and scale
            eeg_raw = np.clip(eeg_raw, -1024, 1024)
            eeg_raw = np.nan_to_num(eeg_raw, nan=0.0) / 32.0
            eeg_raw = eeg_raw.astype(np.float32).T

        # Combination of spec and eeg_tf
        if self.data_type == 'spec+eeg_tf':
            if self.transform:
                spec_image = self.transform['spec'](image=spec_image)['image']
                eeg_tf_image = self.transform['eeg_tf'](image=eeg_tf_image)['image']
            return spec_image, eeg_tf_image, torch.from_numpy(self.targets[idx])
        elif self.data_type == 'spec+eeg_tf+eeg':
            if self.transform:
                spec_image = self.transform['spec'](image=spec_image)['image']
                eeg_tf_image = self.transform['eeg_tf'](image=eeg_tf_image)['image']
                if self.transform['eeg']:
                    eeg_raw = self.transform['eeg'](image=eeg_raw)['image']
            return spec_image, eeg_tf_image, eeg_raw, torch.from_numpy(self.targets[idx])

        # Final image
        if self.data_type == 'spec':
            image = spec_image
        elif self.data_type == 'eeg_tf':
            image = eeg_tf_image
        elif self.data_type == 'eeg':
            image = eeg_raw
        
        if self.transform:
            image = self.transform(image=image)['image']

        return image, torch.from_numpy(self.targets[idx])

def time_crop_mask(image, fill_value=0, max_trim=100, **kwargs):
    trim_samples = random.randint(0, max_trim)
    image[:, :trim_samples, :] = fill_value
    image[:, -trim_samples:, :] = fill_value
    return image

def ch_vertical_flip(image, ch_num=19, p=0.5, **kwargs):
    freq_samples = image.shape[0] // ch_num
    for i in range(ch_num):
        if random.random() < p:
            image[(i-1)*freq_samples:i*freq_samples, ...] = np.flip(image[(i-1)*freq_samples:i*freq_samples, ...], axis=0)
    return image

def get_datasets(CFG, data, df_train, df_validation):
    if CFG.data_type == 'spec' or CFG.data_type == 'eeg_tf':
        transform = {
        'train':
        A.Compose([
            A.CoarseDropout(**CFG.coarse_dropout_args),
            ToTensorV2()
        ]),
        'validation':
        A.Compose([
            ToTensorV2()
        ])}
    elif CFG.data_type == 'spec+eeg_tf' or CFG.data_type == 'spec+eeg_tf+eeg':
        transform = {
        'train':
            {'spec': A.Compose([A.CoarseDropout(p=0.5, max_holes=8, max_height=64, max_width=64), ToTensorV2()]),
             'eeg_tf': A.Compose([A.CoarseDropout(p=0.5, max_holes=8, max_height=128, max_width=128), ToTensorV2()]),
             'eeg': A.Compose([A.XYMasking(p=0.5, num_masks_y=(2, 4), mask_y_length=(500, 2000))])},
        'validation':
            {'spec': A.Compose([ToTensorV2()]), 
             'eeg_tf': A.Compose([ToTensorV2()]),
             'eeg': None}
        }
    elif CFG.data_type == 'eeg':
        transform = {
        'train': #None,
            A.Compose([
                A.XYMasking(**CFG.xymasking_args)
            ]),
        'validation':
            None
        }
    
    train_dataset = HMSDataset(CFG, df=df_train, data=data, transform=transform['train'])
    validation_dataset = HMSDataset(CFG, df=df_validation, data=data, transform=transform['validation'])
    datasets = {'train': train_dataset, 'validation': validation_dataset}
    return datasets

from torch.utils._pytree import tree_flatten, tree_unflatten
class MixUpOneHot(v2.MixUp):
    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)
        needs_transform_list = self._needs_transform_list(flat_inputs)

        labels = self._labels_getter(inputs)
        if not isinstance(labels, torch.Tensor):
            raise ValueError(f"The labels must be a tensor, but got {type(labels)} instead.")

        params = {
            "labels": labels,
            "batch_size": labels.shape[0],
            **self._get_params(
                [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
            ),
        }

        # By default, the labels will be False inside needs_transform_list, since they are a torch.Tensor coming
        # after an image or video. However, we need to handle them in _transform, so we make sure to set them to True
        needs_transform_list[next(idx for idx, inpt in enumerate(flat_inputs) if inpt is labels)] = True
        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)
 
    def _mixup_label(self, label: torch.Tensor, *, lam: float) -> torch.Tensor:
        if not label.dtype.is_floating_point:
            label = label.float()
        return label.roll(1, 0).mul_(1.0 - lam).add_(label.mul(lam))
    
def get_dataloaders(CFG, datasets):
    if CFG.data_type == 'eeg' or CFG.data_type == 'spec+eeg_tf+eeg':
        drop_last = True
    else:
        drop_last = False

    if CFG.use_mixup:
        mixup = MixUpOneHot(alpha=CFG.mixup_alpha, num_classes=CFG.num_classes)
        def collate_fn(batch):
            return mixup(*default_collate(batch))
        train_loader = DataLoader(datasets['train'], batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.dataloader_num_workers, 
                                  collate_fn=collate_fn, pin_memory=CFG.pin_memory, drop_last=drop_last)
        validation_loader = DataLoader(datasets['validation'], batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.dataloader_num_workers, 
                                       pin_memory=CFG.pin_memory, drop_last=drop_last)
    else:
        train_loader = DataLoader(datasets['train'], batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.dataloader_num_workers, 
                                  pin_memory=CFG.pin_memory, drop_last=drop_last)
        validation_loader = DataLoader(datasets['validation'], batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.dataloader_num_workers, 
                                       pin_memory=CFG.pin_memory, drop_last=drop_last)
    dataloaders = {'train': train_loader, 'validation': validation_loader}
    return dataloaders