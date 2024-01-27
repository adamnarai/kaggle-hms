import pandas as pd
import pandas.api.types
import numpy as np

from typing import Optional
import torch
import torch.nn as nn

def kl_divergence(y: pd.DataFrame, y_pred: pd.DataFrame, epsilon: float=10**-15, micro_average: bool=True, sample_weights: Optional[pd.Series]=None):
    # Overwrite y for convenience
    for col in y.columns:
        # Prevent issue with populating int columns with floats
        if not pandas.api.types.is_float_dtype(y[col]):
            y[col] = y[col].astype(float)

        # Clip both the min and max following Kaggle conventions for related metrics like log loss
        # Clipping the max avoids cases where the loss would be infinite or undefined, clipping the min
        # prevents users from playing games with the 20th decimal place of predictions.
        y_pred[col] = np.clip(y_pred[col], epsilon, 1 - epsilon)

        y_nonzero_indices = y[col] != 0
        y[col] = y[col].astype(float)
        y.loc[y_nonzero_indices, col] = y.loc[y_nonzero_indices, col] * np.log(y.loc[y_nonzero_indices, col] / y_pred.loc[y_nonzero_indices, col])
        # Set the loss equal to zero where y_true equals zero following the scipy convention:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html#scipy.special.rel_entr
        y.loc[~y_nonzero_indices, col] = 0

    return np.average(y.mean())

class KLDivLossWithLogits(nn.KLDivLoss):

    def __init__(self):
        super().__init__(reduction="batchmean")

    def forward(self, y, t):
        y = nn.functional.log_softmax(y,  dim=1)
        loss = super().forward(y, t)

        return loss


class KLDivLossWithLogitsForVal(nn.KLDivLoss):
    
    def __init__(self):
        """"""
        super().__init__(reduction="batchmean")
        self.log_prob_list  = []
        self.label_list = []

    def forward(self, y, t):
        y = nn.functional.log_softmax(y, dim=1)
        self.log_prob_list.append(y.numpy())
        self.label_list.append(t.numpy())
        
    def compute(self):
        log_prob = np.concatenate(self.log_prob_list, axis=0)
        label = np.concatenate(self.label_list, axis=0)
        final_metric = super().forward(
            torch.from_numpy(log_prob),
            torch.from_numpy(label)
        ).item()
        self.log_prob_list = []
        self.label_list = []
        
        return final_metric