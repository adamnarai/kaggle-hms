import random
import os
import numpy as np
import torch
import numbers
from PIL import Image
from skimage import color

def seed_everything(seed, deterministic=False, benchmark=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
