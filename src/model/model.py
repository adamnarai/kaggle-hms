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
    
class Wave_Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation_rates: int, kernel_size: int = 3):
        """
        WaveNet building block.
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        :param dilation_rates: how many levels of dilations are used.
        :param kernel_size: size of the convolving kernel.
        """
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True))
        
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=True))
        
        for i in range(len(self.convs)):
            nn.init.xavier_uniform_(self.convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.convs[i].bias)

        for i in range(len(self.filter_convs)):
            nn.init.xavier_uniform_(self.filter_convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.filter_convs[i].bias)

        for i in range(len(self.gate_convs)):
            nn.init.xavier_uniform_(self.gate_convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.gate_convs[i].bias)

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            tanh_out = torch.tanh(self.filter_convs[i](x))
            sigmoid_out = torch.sigmoid(self.gate_convs[i](x))
            x = tanh_out * sigmoid_out
            x = self.convs[i + 1](x) 
            res = res + x
        return res
    
class WaveNet(nn.Module):
    def __init__(self, input_channels: int = 1, kernel_size: int = 3):
        super(WaveNet, self).__init__()
        self.model = nn.Sequential(
                Wave_Block(input_channels, 8, 12, kernel_size),
                Wave_Block(8, 16, 8, kernel_size),
                Wave_Block(16, 32, 4, kernel_size),
                Wave_Block(32, 64, 1, kernel_size) 
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1) 
        return self.model(x)


class WaveNetCustom(nn.Module):
    def __init__(self, eeg_ch):
        super(WaveNetCustom, self).__init__()
        self.model = WaveNet()
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = 0.0
        self.head = nn.Sequential(
            nn.Linear(eeg_ch*64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 6)
        )
        
    def forward(self, x: torch.Tensor):
        bs, t, c = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(bs*c, t, 1)
        x = self.model(x)
        x = self.global_avg_pooling(x)
        x = x.reshape(bs, -1)

        # z_list = []
        # for i in range(x.shape[-1]):
        #     x1 = self.model(x[:, :, i:i+1])
        #     x1 = self.global_avg_pooling(x1)
        #     x1 = x1.squeeze()
        #     z_list.append(x1)
        # print(x1.shape)
        
        # x = torch.cat(z_list, dim=1)
        # print(x.shape)

        return self.head(x)
    