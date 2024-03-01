import os
import numpy as np
import random
from functools import partial

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
                spec_image = self.transform(image=spec_image)['image']
                eeg_tf_image = self.transform(image=eeg_tf_image)['image']
            return spec_image, eeg_tf_image, torch.from_numpy(self.targets[idx])

        # Final image
        if self.data_type == 'spec':
            image = spec_image
        elif self.data_type == 'eeg_tf':
            image = eeg_tf_image
        
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
    transform = {
    'train':
    A.Compose([
        # A.RingingOvershoot(**CFG.ringing_overshoot_args),
        # A.MedianBlur(**CFG.median_blur_args),
        # A.Sharpen(**CFG.sharpen_args),
        # A.XYMasking(**CFG.xy_masking_args),
        # A.RandomGamma(),
        # A.MultiplicativeNoise(),
        # A.RandomBrightnessContrast(),
        # A.PixelDropout(drop_value=0.5),
        # A.VerticalFlip(p=CFG.vertical_flip_p),
        # A.Lambda(image=partial(ch_vertical_flip, **CFG.ch_vertical_flip_args)),
        A.CoarseDropout(**CFG.coarse_dropout_args),
        # A.Lambda(image=partial(time_crop_mask, **CFG.time_crop_args), p=CFG.time_crop_p),
        ToTensorV2()
    ]),
    'validation':
     A.Compose([
        ToTensorV2()
    ])}
    
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
    if CFG.use_mixup:
        mixup = MixUpOneHot(alpha=CFG.mixup_alpha, num_classes=CFG.num_classes)
        def collate_fn(batch):
            return mixup(*default_collate(batch))
        train_loader = DataLoader(datasets['train'], batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.dataloader_num_workers, collate_fn=collate_fn, pin_memory=True)
        validation_loader = DataLoader(datasets['validation'], batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.dataloader_num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(datasets['train'], batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.dataloader_num_workers, pin_memory=True)
        validation_loader = DataLoader(datasets['validation'], batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.dataloader_num_workers, pin_memory=True)
    dataloaders = {'train': train_loader, 'validation': validation_loader}
    return dataloaders