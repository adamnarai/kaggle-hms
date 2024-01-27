import torch.nn as nn
import torch.nn.functional as F
import timm

class SpecCNN(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True, in_channels=None):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_channels)
    
    def forward(self, x):
        return self.model(x)