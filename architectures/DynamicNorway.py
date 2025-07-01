import torch
import torch.nn as nn

# Series configuration of DynamicNorway

INPUT_CHANNELS = 9

class SeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SeparableBlock, self).__init__()

        # Depthwise Convolutions
        self.depthwise = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        )
        
        # Pointwise Convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        depthwise_out = self.depthwise(x)
        pointwise_out = self.pointwise(depthwise_out) 
        return pointwise_out
        
class BlockWithSkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockWithSkipConnection, self).__init__()

        self.separable_blocks = nn.Sequential(
            SeparableBlock(in_channels, out_channels),
            SeparableBlock(out_channels, out_channels),
            SeparableBlock(out_channels, out_channels)
        )

        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        block_out = self.separable_blocks(x)
        skip_connection_out = self.skip_connection(x)
        return block_out + skip_connection_out

class DynamicNorway(nn.Module):
    def __init__(self, b=2, m=1.5, num_classes=8):
        super(DynamicNorway, self).__init__()

        self.b = b
        self.m = m

        # Initial convolution (64 filters, 3x3)
        self.initial_conv = nn.Conv2d(INPUT_CHANNELS, 64, kernel_size=3, padding=1)
        
        self.Bblocks = nn.ModuleList()
        
        in_channels = 64
        
        for i in range(1, b + 1):  # Repeat b times
            # The output channels for the separable blocks are determined as Cᵢ = 64 * m^i
            out_channels = int(64 * (m ** i))

            self.Bblocks.append(BlockWithSkipConnection(in_channels, out_channels))
            
            in_channels = out_channels 

        
        # Skip connection convolution (Conv 48, 3×3 in diagram)
        self.skip_connection = nn.Conv2d(64, 48, kernel_size=3, padding=1)
        
        in_channels += 48  # Add the skip connection channels to the total input channels for the decoder
        out_channels = int(64 * (m ** b))  # C_b = 64 * m^b
    
        # Add a separable block to the decoder
        self.decoder_block = BlockWithSkipConnection(in_channels, out_channels)

        # Final 1x1 conv to get the final output
        self.final_conv = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.initial_conv(x)  

        identity = self.skip_connection(x)
        
        for block in self.Bblocks:
            x = block(x)    

        x = torch.cat([x, identity], dim=1) 

        x = self.decoder_block(x)
        
        x = self.final_conv(x) 
        return x


