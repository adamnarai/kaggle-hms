import torch
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
    
class SpecTfCNN(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        if isinstance(model_name, str) == 1:
            model_name_spec = model_name_eeg_tf = model_name
        elif isinstance(model_name, tuple) and len(model_name) == 2:
            model_name_spec, model_name_eeg_tf = model_name
        self.num_classes = num_classes
        self.model_spec = timm.create_model(model_name=model_name_spec, pretrained=pretrained, num_classes=128, in_chans=1)
        self.model_eeg_tf = timm.create_model(model_name=model_name_eeg_tf, pretrained=pretrained, num_classes=128, in_chans=1)
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x1, x2):
        x = torch.cat((self.model_spec(x1), self.model_eeg_tf(x2)), dim=1)
        return self.classifier(x)
    
class SpecTfCNN_v2(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        if isinstance(model_name, str) == 1:
            model_name_spec = model_name_eeg_tf = model_name
        elif isinstance(model_name, tuple) and len(model_name) == 2:
            model_name_spec, model_name_eeg_tf = model_name
        self.num_classes = num_classes
        self.model_spec = timm.create_model(model_name=model_name_spec, pretrained=pretrained, num_classes=128, in_chans=1)
        self.model_eeg_tf = timm.create_model(model_name=model_name_eeg_tf, pretrained=pretrained, num_classes=128, in_chans=1)
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x1, x2):
        x = F.relu(torch.cat((self.model_spec(x1), self.model_eeg_tf(x2)), dim=1))
        return self.classifier(x)