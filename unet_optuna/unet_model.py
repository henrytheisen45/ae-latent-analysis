import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive convolutional layers with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self._check_image_pooling_compatibility()

    def forward(self, x):
        return self.double_conv(x)


class FlexibleUNetAutoencoder(nn.Module):
    """
    Flexible UNet-style Autoencoder that adapts to different image sizes and architectures
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        img_size: Input image size (assumes square images)
        base_channels: Base number of channels (will be multiplied at each level)
        num_levels: Number of encoder/decoder levels (depth of the network)
        kernel_size: Kernel size for convolutions
        padding: Padding for convolutions
        pooling_size: Size of pooling/upsampling operations
    """
    def __init__(self, in_channels, img_size, base_channels=64, num_levels=3, 
                 kernel_size=3, padding=1, pooling_size=2):
        super().__init__()
        
        self.in_channels = in_channels
        self.img_size = img_size
        self.base_channels = base_channels
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.padding = padding
        self.pooling_size = pooling_size

        # Check img size compatibility
        # TODO change dataloader to pad if not compatible
        self._check_image_pooling_compatibility()
        
        # Build encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        current_channels = in_channels
        for i in range(num_levels):
            out_channels = base_channels * (2 ** i)
            self.encoders.append(DoubleConv(current_channels, out_channels, kernel_size, padding))
            self.pools.append(nn.MaxPool2d(pooling_size))
            current_channels = out_channels

        
        # Bottleneck
        bottleneck_channels = base_channels * (2 ** num_levels)
        self.bottleneck = DoubleConv(current_channels, bottleneck_channels, kernel_size, padding)
        
        # Build decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        current_channels = bottleneck_channels
        for i in range(num_levels - 1, -1, -1):
            out_channels = base_channels * (2 ** i)
            self.upconvs.append(
                nn.ConvTranspose2d(current_channels, out_channels, 
                                  kernel_size=pooling_size, stride=pooling_size)
            )
            # *2 because of skip connection concatenation
            self.decoders.append(DoubleConv(out_channels * 2, out_channels, kernel_size, padding))
            current_channels = out_channels
        
        # Final output layer
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        
        # Calculate expected bottleneck size
        self.bottleneck_size = self._calculate_bottleneck_size()
    

        
    def _check_image_pooling_compatibility(self):
        expected_divisor = self.pooling_size ** self.num_levels
        if self.img_size % expected_divisor != 0:
            warnings.warn(
                f"img_size={self.img_size} is not divisible by {expected_divisor} "
                f"(pooling_size={self.pooling_size}^{self.num_levels}). "
                "Decoder will interpolate to align skip connections.",
                UserWarning
            )

    def _calculate_bottleneck_size(self):
        """Calculate the spatial size at the bottleneck"""
        size = self.img_size
        for _ in range(self.num_levels):
            size = size // self.pooling_size
        return size
        
    
    def encode(self, x):
        """Encode input to latent space"""
        skip_connections = []
        
        # Pass through encoder levels
        for i in range(self.num_levels):
            x = self.encoders[i](x)
            skip_connections.append(x)
            x = self.pools[i](x)
        
        # Bottleneck (latent space)
        latent = self.bottleneck(x)
        
        return latent, skip_connections
    
    def decode(self, latent, skip_connections):
        """Decode from latent space to output"""
        x = latent
        
        # Reverse the skip connections for decoder
        skip_connections = skip_connections[::-1]
        
        # Pass through decoder levels
        for i in range(self.num_levels):
            x = self.upconvs[i](x)
            
            # Handle potential size mismatches with skip connections
            skip = skip_connections[i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)
        
        # Final adjustment to ensure output matches input size
        if x.shape[2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        x = self.out_conv(x)
        return torch.sigmoid(x)
    
    def forward(self, x, return_latent=False):
        """Forward pass through the autoencoder"""
        latent, skip_connections = self.encode(x)
        reconstruction = self.decode(latent, skip_connections)
        
        if return_latent:
            return reconstruction, latent
        return reconstruction
    
    def get_latent_vector(self, x):
        """Get flattened latent representation for a batch of images"""
        latent, _ = self.encode(x)
        # Flatten the spatial dimensions: (batch, channels, h, w) -> (batch, channels*h*w)
        return latent.view(latent.size(0), -1)


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
