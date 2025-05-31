import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth
from einops import rearrange

class ConvNeXtStem(nn.Module):
    """Downsamples input to match _center_embed's spatial dimensions."""
    def __init__(self, in_chans=3, out_chans=96, kernel_size=3, stride=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chans, out_chans, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=kernel_size//2  # Maintains exact 48→16 downsampling
        )
        self.norm = nn.LayerNorm(out_chans, eps=1e-6)
        
    def forward(self, x):
        # Input: [B, 3, 48, 48] → Output: [B, 96, 16, 16]
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)  # [B, C, H, W]

# V0
# class ConvNeXtBlock(nn.Module):
# Maintains 16x16 resolution for coronary CTA features.
    # def __init__(self, dim, drop_path=0.1): # avoid overfitting
    #     super().__init__()
    #     self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
    #     self.norm = nn.LayerNorm(dim, eps=1e-6)
    #     self.pw_conv = nn.Sequential(
    #         nn.Linear(dim, 4 * dim),
    #         nn.GELU(),
    #         nn.Linear(4 * dim, dim)
    #     )
    #     self.drop_path = StochasticDepth(drop_path, mode="row")
    #     self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1) * 1e-6)
        
    # def forward(self, x):
    #     identity = x
    #     x = self.dw_conv(x)
    #     x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
    #     x = self.norm(x)
    #     x = self.pw_conv(x)
    #     x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
    #     return identity + self.drop_path(x * self.gamma)


class ConvNeXt(nn.Module):
    """Produces [batch, slices, 16, 16] output matching _center_embed."""
    def __init__(self, in_chans=3, num_slices=196, dim=96, depths=[3]):
        super().__init__()
        # Stem: 48x48 → 16x16
        self.stem = ConvNeXtStem(in_chans=in_chans, out_chans=dim)
        
        # Main blocks
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(dim, drop_path=0.1)
            for _ in range(depths[0])
        ])
        
        # Final projection to match slice dimensions
        self.head = nn.Conv2d(dim, 1, kernel_size=1)  # [B, 1, 16, 16]

    def forward(self, x):
        # Input: [B*num_slices, 3, 48, 48]
        x = self.stem(x)    # [B*196, 96, 16, 16]
        x = self.blocks(x)  # [B*196, 96, 16, 16]
        x = self.head(x)    # [B*196, 1, 16, 16]
        return x.squeeze(1) # [B*196, 16, 16]
"""
# V1  Enhanced ConvNeXt Block with Hybrid Attention
class ConvNeXtBlock(nn.Module):
     def __init__(self, dim, drop_path=0.1):
         super().__init__()
         # Depthwise Conv + Channel Attention
         self.dw_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
         self.ca = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),  # -> [B, dim, 1, 1]
             nn.Flatten(start_dim=1),   # -> [B, dim]
             nn.Linear(dim, dim//4),
             nn.GELU(),
             nn.Linear(dim//4, dim),
             nn.Sigmoid()
         )
       
         # Pixel Attention
         self.pa = nn.Sequential(
             nn.Conv2d(dim, 1, 1),
             nn.Sigmoid()
         )
         # FFN with Gated Mechanism
         self.ffn = nn.Sequential(
             nn.LayerNorm(dim),
             nn.Linear(dim, 4*dim),
             nn.GELU(),
             nn.Linear(4*dim, dim)
         )
       
         self.drop_path = StochasticDepth(drop_path, mode="row")
         self.gamma = nn.Parameter(torch.ones(1)*1e-6)
     def forward(self, x):
         # x shape: [B, dim, H, W]
         identity = x
       
         # Depthwise
         x = self.dw_conv(x)
         # Channel Attention
         B, C, H, W = x.shape
         ca_weight = self.ca(x)                # shape: [B, C]
         ca_weight = ca_weight.view(B, C, 1, 1) # -> [B, C, 1, 1]
         x = x * ca_weight                     # broadcast multiply
       
         # Pixel Attention
         pa_weight = self.pa(x)
         x = x * pa_weight
       
         # FFN with residual
         x = x.permute(0,2,3,1)
         x = self.ffn(x) * self.gamma  # Gated scale
         x = x.permute(0,3,1,2)
       
         return identity + self.drop_path(x)
"""
# V2  Enhanced ConvNeXt Block with Hybrid Attention and Gated FFN
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.1):
        super().__init__()
        # Depthwise conv with elongated kernels
        self.dw_conv = nn.Conv2d(dim, dim, (7,3), padding=(3,1), groups=dim)
        
        # Vessel-aware attention
        self.va_attn = nn.Sequential(
            nn.Conv2d(dim, 1, 3, padding=1),  # Local calcification detector
            nn.Sigmoid()
        )
        
        # Enhanced FFN with per-channel scaling
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim)
        )
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1) * 1e-6)
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x):
        identity = x
        
        # Anisotropic depthwise (vertical emphasis)
        x = self.dw_conv(x)
        
        # Vessel attention
        attn = self.va_attn(x)  # [B,1,H,W]
        x = x * attn + identity  # Preserve original features
        
        # FFN
        x = x.permute(0,2,3,1)
        x = self.ffn(x)
        x = x.permute(0,3,1,2)
        
        return identity + self.drop_path(x * self.gamma)