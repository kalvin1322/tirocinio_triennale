import torch
import torch.nn as nn

class ThreeL_SSNet(nn.Module):
    def __init__(self):
        super(ThreeL_SSNet, self).__init__()
        # Define your model architecture here
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))