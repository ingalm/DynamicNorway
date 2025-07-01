import torch.nn as nn
from monai.networks.nets import UNet

# A U-Net class that uses MONAI's UNet implementation.
# Created for integration with the LCC4CD pipeline in this repository.

class CustomUNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomUNet, self).__init__()
        self.model = UNet(
            spatial_dims=2,
            in_channels=9,
            out_channels=num_classes,
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm='batch',
        )

    def forward(self, x):
        return self.model(x)
