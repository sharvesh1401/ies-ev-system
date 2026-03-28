"""
TCN-Transformer Architecture for Teacher and Student Energy Predictors.
"""

import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

    def forward(self, x):
        out = torch.relu(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)


class TeacherModel(nn.Module):
    def __init__(self, seq_len=64, n_features=14):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        
        # TCN
        self.tcn = nn.Sequential(
            TCNBlock(n_features, 32, dilation=1, dropout=0.1),
            TCNBlock(32, 64, dilation=2, dropout=0.1),
            TCNBlock(64, 128, dilation=4, dropout=0.1),
        )
        
        # Projection
        self.proj = nn.Linear(128, 256)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=1024,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Output heads
        self.soc_head = nn.Linear(256, 2)
        self.soh_head = nn.Linear(256, 2)
        self.energy_head = nn.Linear(256, 2)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, self.n_features)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        
        soc_out = self.soc_head(x)
        soh_out = self.soh_head(x)
        energy_out = self.energy_head(x)
        
        return (soc_out[:, 0], soc_out[:, 1], 
                soh_out[:, 0], soh_out[:, 1],
                energy_out[:, 0], energy_out[:, 1])


class StudentModel(nn.Module):
    def __init__(self, seq_len=64, n_features=14):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        
        # TCN (smaller)
        self.tcn = nn.Sequential(
            TCNBlock(n_features, 16, dilation=1, dropout=0.1),
            TCNBlock(16, 32, dilation=2, dropout=0.1),
            TCNBlock(32, 64, dilation=4, dropout=0.1),
        )
        
        # Projection
        self.proj = nn.Linear(64, 128)
        
        # Transformer (smaller)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output heads
        self.soc_head = nn.Linear(128, 2)
        self.soh_head = nn.Linear(128, 2)
        self.energy_head = nn.Linear(128, 2)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, self.n_features)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        
        soc_out = self.soc_head(x)
        soh_out = self.soh_head(x)
        energy_out = self.energy_head(x)
        
        return (soc_out[:, 0], soc_out[:, 1], 
                soh_out[:, 0], soh_out[:, 1],
                energy_out[:, 0], energy_out[:, 1])
