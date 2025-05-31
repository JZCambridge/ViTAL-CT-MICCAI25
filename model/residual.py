import torch
import torch.nn as nn

class SimpleResNet(nn.Module):
    def __init__(self, input_channels=3, output_size=256):
        super().__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolutional layer (residual path)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Shortcut connection (identity if input channels match, otherwise adjust)
        self.shortcut = nn.Sequential()
        if input_channels != 64:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=1),
                nn.BatchNorm2d(64)
            )
        
        # Final layers to produce output of size [batch, output_size]
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual  # Residual connection
        out = self.relu(out)
        
        out = self.avgpool(out)  # Global average pooling
        out = torch.flatten(out, 1)
        out = self.fc(out)       # Final linear layer
        
        return out