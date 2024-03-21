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
                Wave_Block(input_channels, 16, 8, kernel_size),
                nn.AvgPool1d(10),
                Wave_Block(16, 32, 5, kernel_size),
                nn.AvgPool1d(10),
                Wave_Block(32, 64, 3, kernel_size),
                nn.AvgPool1d(10),
                Wave_Block(64, 64, 2, kernel_size),
                nn.AvgPool1d(2)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1) 
        return self.model(x)


class WaveNetCustom(nn.Module):
    def __init__(self, num_classes, eeg_ch, dropout=0.0, hidden_features=64, headless=False):
        super(WaveNetCustom, self).__init__()
        self.model = WaveNet()
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = dropout
        self.hidden_features = hidden_features
        self.head = nn.Sequential(
            nn.Linear(eeg_ch*64, self.hidden_features),
            nn.BatchNorm1d(self.hidden_features),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_features, num_classes)
        )
        self.lstm = nn.LSTM(64, 64, 1)
        self.lstm_dropout = nn.Dropout(0.5)
        self.headless = headless
        
    def forward(self, x: torch.Tensor):
        bs, t, c = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(bs*c, t, 1)
        x = self.model(x)

        # LSTM
        x = x.permute(0, 2, 1)
        x, (h_n, c_n) = self.lstm(x)
        x = self.lstm_dropout(x)
        x = x.permute(0, 2, 1)

        x = self.global_avg_pooling(x)
        x = x.reshape(bs, -1)

        if not self.headless:
            return self.head(x)
        else:
            return x
    
class SpecTfEEGCNN(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True, eeg_ch=18, dropout=0.0, hidden_features=64):
        super().__init__()
        if isinstance(model_name, str) == 1:
            model_name_spec = model_name_eeg_tf = model_name
        elif isinstance(model_name, tuple) and len(model_name) == 2:
            model_name_spec, model_name_eeg_tf = model_name
        self.num_classes = num_classes
        self.model_spec = timm.create_model(model_name=model_name_spec, pretrained=pretrained, num_classes=128, in_chans=1)
        self.model_eeg_tf = timm.create_model(model_name=model_name_eeg_tf, pretrained=pretrained, num_classes=128, in_chans=1)
        self.model_eeg = WaveNetCustom(num_classes=num_classes, eeg_ch=eeg_ch, dropout=dropout, hidden_features=hidden_features, headless=True)
        self.eeg_head = nn.Sequential(
            nn.Linear(eeg_ch*64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.classifier = nn.Linear(3*128, num_classes)
    
    def forward(self, x1, x2, x3):
        x = torch.cat((self.model_spec(x1), self.model_eeg_tf(x2), self.eeg_head(self.model_eeg(x3))), dim=1)
        return self.classifier(x)
    
class WaveNetCustom2(nn.Module):
    def __init__(self, num_classes, eeg_ch, dropout=0.0, hidden_features=64):
        super(WaveNetCustom2, self).__init__()
        self.model = nn.Sequential(
            Wave_Block(eeg_ch, 16, 8, 3),
            nn.AvgPool1d(5),
            Wave_Block(16, 32, 5, 3),
            nn.AvgPool1d(5),
            Wave_Block(32, 64, 3, 3),
            nn.AvgPool1d(5),
            Wave_Block(64, 64, 2, 3)
            # nn.AvgPool1d(5),
        )
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = dropout
        self.hidden_features = hidden_features
        self.head = nn.Sequential(
            nn.Linear(64, self.hidden_features),
            nn.BatchNorm1d(self.hidden_features),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_features, num_classes)
        )
        
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1) # (bs, t, c) -> (bs, c, t)
        x = self.model(x)
        x = self.global_avg_pooling(x).squeeze()

        return self.head(x)
    