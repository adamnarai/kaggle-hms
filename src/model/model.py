import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from torchvision import models

class SpecCNN(nn.Module):
    def __init__(self, model_name, num_classes, weights='DEFAULT'):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = models.get_model(model_name, weights=weights)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace classifier layer (unfreezed)
        self._replace_classifier()
    
    def forward(self, x):
        return self.model(x)

    def _replace_classifier(self):
        if self.model_name.startswith('resnet'):
            self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.num_classes)
        elif self.model_name.startswith('efficientnet'):
            self.model.classifier[1] = nn.Linear(in_features=self.model.classifier[1].in_features, out_features=self.num_classes, bias=True)
        elif self.model_name.startswith('convnext'):
            self.model.classifier[2] = nn.Linear(in_features=self.model.classifier[2].in_features, out_features=self.num_classes, bias=True)
        elif self.model_name.startswith('vit'):
            self.model.heads.head = nn.Linear(in_features=self.model.heads.head.in_features, out_features=self.num_classes, bias=True)
        elif self.model_name.startswith('swin'):
            self.model.head = nn.Linear(in_features=self.model.head.in_features, out_features=self.num_classes, bias=True)
        else:
            raise NotImplementedError('Classifier layer replacement not implemented for this model')
        return