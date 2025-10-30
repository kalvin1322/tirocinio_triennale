import torch
import torch.nn as nn

class UNet_V1(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_encoders: int = 3, start_middle_channels: int = 32):
        super().__init__()
        self.num_encoders = num_encoders
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU()
            )
        
        # --- Encoder ---
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        for i in range(num_encoders):
            if i == 0:
                self.encoders.append(conv_block(in_channels, start_middle_channels))
            else:
                self.encoders.append(conv_block(start_middle_channels * (2 ** (i - 1)), start_middle_channels * (2 ** i)))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # --- Bottleneck ---
        self.bottleneck = conv_block(start_middle_channels * (2 ** (num_encoders - 1)), start_middle_channels * (2 ** num_encoders))

        # --- Decoder ---
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(num_encoders - 1, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(start_middle_channels * (2 ** (i + 1)), start_middle_channels * (2 ** i), kernel_size=2, stride=2))
            self.decoders.append(conv_block(start_middle_channels * (2 ** (i + 1)), start_middle_channels * (2 ** i)))

        # --- Output ---
        self.outconv = nn.Conv2d(start_middle_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for i in range(self.num_encoders):
            x = self.encoders[i](x)
            skip_connections.append(x)
            x = self.pools[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)

        # Decoder 
        skip_connections = skip_connections[::-1]  
        
        for i in range(self.num_encoders):
            x = self.upconvs[i](x)
            x = torch.cat([x, skip_connections[i]], dim=1)
            x = self.decoders[i](x)

        # Output
        out = self.outconv(x)
        return out