import torch
import torch.nn as nn
from .ConvNeXt import ConvNeXt
from .resnet import resnet18
from .vit_small_simplify import VisionTransformer_posembed as VisionTransformer
from einops import rearrange
import math
from torch.utils import model_zoo
from torch import nn
import re
from collections import OrderedDict


class Res_ViT(nn.Module):
    def __init__(self, patch_size = 16, embed_dim = 768, depth = 12, num_heads = 12, res_bs = 196, num_classes = 1000, num_slices=196):
        super().__init__()
        self.res_bs = res_bs
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.vit = VisionTransformer(patch_size = patch_size, 
                                        embed_dim = embed_dim, 
                                        depth = depth, 
                                        num_heads = num_heads, 
                                        num_classes = num_classes,
                                        drop_rate=0.001,          # General dropout
                                        pos_drop_rate=0.001,      # Positional embedding dropout
                                        patch_drop_rate=0,    # Patch dropout (aggressive for CTA)
                                        proj_drop_rate=0,     # Projection dropout
                                        attn_drop_rate=0.001,     # Attention dropout
                                        drop_path_rate=0.001,     # Stochastic depth
                                        )

        self.linear = nn.Linear(3*patch_size**2, patch_size**2)
        self.norm = nn.BatchNorm3d(1)
        self.leaky_relu = nn.LeakyReLU(0.1)

        # unet
        self.unet_norm = nn.LayerNorm([patch_size, patch_size])
        ## attention + linear layer
        self.unet_proj = nn.Linear(16*16, 16*16)  # Lightweight
        # self.unet_attn = nn.Conv3d(1, 1, kernel_size=1)  # Spatial gating
        ## attention layer only
        self.unet_attn = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=3, padding=1),  # Process UNet embeddings
            nn.Sigmoid()
        )
        ## linear only 
        # self.unet_proj = nn.Sequential(
        #     nn.Linear(16*16, 32),  # Reduce dimensionality
        #     nn.ReLU(),
        #     nn.Linear(32, 16*16)
        # )

        # Replace ResNet with ConvNeXt
        self.residual = ConvNeXt(
            in_chans=3,
            num_slices=num_slices,
            dim=96,
            depths=[3]  # 3 ConvNeXt blocks
        )
        #self.residual = resnet18(num_classes = 256)

    def _resnet_embed(self, x, bs):
        b, h, w, l = x.shape  # [32, 48, 48, 196]
        
        # Reshape to [32*196, 3, 48, 48]
        x = rearrange(x, 'b h w l -> (b l) h w')
        x = x.unsqueeze(1).expand(-1, 3, -1, -1)
        
        # Process through ConvNeXt
        out = self.residual(x)  # [B*196, 16, 16]

        #out = rearrange(out, 'b (h w) -> b h w', h = h//3, w = w//3) #for resnet

        
        # Reshape to match _center_embed's [32, 196, 16, 16]
        out = rearrange(out, '(b l) h w -> b l h w', b=b)
        return out
    
    def _center_embed(self,x):
        b, h, w, l = x.shape
        x = x[:, int(h/3):int(2*h/3), int(h/3):int(2*h/3), :]
        x = rearrange(x, 'b h w l -> b l h w')
        return x
    
    def _segment_embed(self,x):
        b, h, w, l = x.shape
        return torch.ones(b, l, h//3, w//3, device = x.device)
    
    def _merge_token(self, tka, tkb, tkc):
        tka_use = tka.unsqueeze(1)
        tkb_use = tkb.unsqueeze(1)

        # Unet attn + linear
        # tkc_flat = tkc.flatten(2)  # [B, L, 256]
        # tkc_proj = self.unet_proj(tkc_flat).view_as(tkc)  # Project
        # tkc_attn = torch.sigmoid(self.unet_attn(tkc_proj.unsqueeze(1)))  # [B, 1, L, H, W]
        # tkc_use = tkc_proj.unsqueeze(1)# * tkc_attn  # Apply mask

        # Unet attn only
        tkc_attn = self.unet_attn(tkc.unsqueeze(1))  # [B, 1, L, H, W]
        tkc_use = tkc.unsqueeze(1) * tkc_attn  # Apply mask only to UNet

        # Linear only
        #tkc_proj = self.unet_proj(tkc.flatten(2)).view_as(tkc)  # [B, L, H, W]
        #tkc_use = tkc_proj.unsqueeze(1)

        mdata = torch.cat((tka_use, tkb_use, tkc_use), dim=1)
        
        b, c, l, h, w = mdata.shape
        mdata = rearrange(mdata, 'b c l h w-> b c h w l')
        outmat = torch.zeros(b, 3, int(math.sqrt(h*w*l)), int(math.sqrt(h*w*l)), device = mdata.device)
        count = int(math.sqrt(h*w*l) / self.patch_size)
        for i in range(l):
            outmat[:,:,self.patch_size*(i//count):self.patch_size*(i//count+1),self.patch_size*(i%count):self.patch_size*(i%count+1)] = mdata[:,:,:,:,i].squeeze() # TODO: 顺序是否正确
        return outmat

    def forward(self, x, pos, unetembed):
        resnet_embed = self._resnet_embed(x,self.res_bs)
        center_embed = self._center_embed(x)
        segment_embed = rearrange(unetembed, 'b l (h w) -> b l h w', h = x.shape[1]//3, w = x.shape[2]//3)
        segment_embed =self.unet_norm(segment_embed)
        x_merge = self._merge_token(center_embed, resnet_embed, segment_embed)
        # x_merge = self._merge_token(center_embed, center_embed, center_embed) # Replicating the only ViT model from benchmark
        # x_merge = self._merge_token(center_embed, segment_embed, segment_embed) # remove the resnet_embed to test the performance of the VIT model
        # x_merge = self._merge_token(center_embed, center_embed+resnet_embed, center_embed+segment_embed) # remove the resnet_embed to test the performance of the VIT model
        x_out = self.vit(x_merge, pos)
        return x_out

