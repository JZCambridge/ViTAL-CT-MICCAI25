# Sheng Wang at Feb 22 2023

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor
from torch.nn.parameter import Parameter

from .base_vit import ViT


class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module, r: int, alpha: int):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b
        self.r = r
        self.alpha = alpha

    def forward(self, x):
        x = self.w(x) + (self.alpha // self.r) * self.w_b(self.w_a(x))
        return x


class LoRA_ViT(nn.Module):
    """Applies low-rank adaptation to a vision transformer.

    Args:
        vit_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, vit_model: ViT, r: int, alpha: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_ViT, self).__init__()

        assert r > 0
        assert alpha > 0
        base_vit_dim = vit_model.transformer.blocks[0].attn.proj_q.in_features
        dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.transformer.blocks)))
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.transformer.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_q_linear = blk.attn.proj_q
            w_v_linear = blk.attn.proj_v
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.proj_q = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q, r, alpha)
            blk.attn.proj_v = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v, r, alpha)

        self.reset_parameters()
        self.lora_vit = vit_model
        if num_classes > 0:
            self.lora_vit.fc = nn.Linear(vit_model.fc.in_features, num_classes)

    def save_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.fc.in_features
        _out = self.lora_vit.fc.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.fc.weight}
        save_file(fc_tensors, filename)

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")
        _in = self.lora_vit.fc.in_features
        _out = self.lora_vit.fc.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.fc.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        _in = self.lora_vit.fc.in_features
        _out = self.lora_vit.fc.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.fc.weight}
        
        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
                
            _in = self.lora_vit.fc.in_features
            _out = self.lora_vit.fc.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.fc.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)


class _LoRA_qkv_timm(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        r: int,
        alpha: int
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)
        self.r = r
        self.alpha = alpha

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += (self.alpha // self.r) * new_q
        qkv[:, :, -self.dim :] += (self.alpha // self.r) * new_v
        return qkv


class LoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: timm_ViT, r: int, alpha: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_ViT_timm, self).__init__()

        assert r > 0
        assert alpha > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))

        # dim = vit_model.head.in_features
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv_timm(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                r,
                alpha
            )
        self.reset_parameters()
        self.lora_vit = vit_model
        self.proj_3d = nn.Linear(num_classes * 30, num_classes)
        if num_classes > 0:
            self.lora_vit.reset_classifier(num_classes=num_classes)
            # self.lora_vit.head = nn.Linear(
            #     self.dim, num_classes)

    def save_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        save_file(fc_tensors, filename)

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        
        save both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        
        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\
            
        load both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
                
            _in = self.lora_vit.head.in_features
            _out = self.lora_vit.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)


class LoRA_ViT_timm_posembed(nn.Module):
    def __init__(self, vit_model: timm_ViT, r: int, alpha: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_ViT_timm_posembed, self).__init__()

        assert r > 0
        assert alpha > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))

        # dim = vit_model.head.in_features
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv_timm(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                r,
                alpha
            )
        self.reset_parameters()
        self.lora_vit = vit_model
        self.proj_3d = nn.Linear(num_classes * 30, num_classes)
        if num_classes > 0:
            self.lora_vit.reset_classifier(num_classes=num_classes)
            # self.lora_vit.head = nn.Linear(
            #     self.dim, num_classes)

    def save_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        save_file(fc_tensors, filename)

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        
        save both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        
        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\
            
        load both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
                
            _in = self.lora_vit.head.in_features
            _out = self.lora_vit.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        return self.lora_vit(x, pos)

    # def forward(self, x: Tensor) -> Tensor:
    #     x = rearrange(x, "b s c h w -> (b s) c h w", s=30)
    #     x = self.lora_vit(x)
    #     x = rearrange(x, "(b s) d -> b (s d)", s=30)
    #     x = self.proj_3d(x)
    #     return x
    
    
class _LoRA_qkv_timm_x(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_qs,
        linear_b_qs,
        linear_a_vs,
        linear_b_vs,
        scale_list,
    ):
        super().__init__()
        self.qkv = qkv
        for i in range(len(linear_a_qs)):
            setattr(self, f'linear_a_q_{i}', linear_a_qs[i])
            setattr(self, f'linear_b_q_{i}', linear_b_qs[i])
            setattr(self, f'linear_a_v_{i}', linear_a_vs[i])
            setattr(self, f'linear_b_v_{i}', linear_b_vs[i])
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)
        self.lora_id = 0
        self.scale_list = scale_list
    
    def change_lora(self, num):
        self.lora_id = num

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        linear_a_q = getattr(self, f'linear_a_q_{self.lora_id}')
        linear_b_q = getattr(self, f'linear_b_q_{self.lora_id}')
        linear_a_v = getattr(self, f'linear_a_v_{self.lora_id}')
        linear_b_v = getattr(self, f'linear_b_v_{self.lora_id}')
        new_q = linear_b_q(linear_a_q(x))
        new_v = linear_b_v(linear_a_v(x))
        qkv[:, :, : self.dim] += self.scale_list[self.lora_id] * new_q
        qkv[:, :, -self.dim :] += self.scale_list[self.lora_id] * new_v
        return qkv
        
class LoRA_ViT_timm_x(nn.Module):
    def __init__(self, vit_model: timm_ViT, lora_files: list, lora_layer=None):
        super(LoRA_ViT_timm_x, self).__init__()

        self.lora_layer = list(range(len(vit_model.blocks)))

        # dim = vit_model.head.in_features
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        
        self.fc_loras = []
        self.num_classes = []

        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False
        
        self.lora_vit = vit_model

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            
            w_a_linear_qs = []
            w_b_linear_qs = []
            w_a_linear_vs = []
            w_b_linear_vs = []
            scale_list = []
            for file_path in lora_files:
                with safe_open(file_path, framework="pt") as f:
                    melo_info = file_path.split("/")[-1].split("_")
                    
                    r = int(melo_info[3])
                    alpha = int(melo_info[4])
                    scale_list.append(alpha // r)
                    
                    w_a_linear_q = nn.Linear(self.dim, r, bias=False)
                    w_b_linear_q = nn.Linear(r, self.dim, bias=False)
                    w_a_linear_v = nn.Linear(self.dim, r, bias=False)
                    w_b_linear_v = nn.Linear(r, self.dim, bias=False)
                    
                    w_a_linear_q.weight = Parameter(f.get_tensor(f"w_a_{t_layer_i * 2:03d}"))
                    w_b_linear_q.weight = Parameter(f.get_tensor(f"w_b_{t_layer_i * 2:03d}"))
                    w_a_linear_v.weight = Parameter(f.get_tensor(f"w_a_{t_layer_i * 2 + 1:03d}"))
                    w_b_linear_v.weight = Parameter(f.get_tensor(f"w_b_{t_layer_i * 2 + 1:03d}"))
                    
                    w_a_linear_qs.append(w_a_linear_q)
                    w_b_linear_qs.append(w_b_linear_q)
                    w_a_linear_vs.append(w_a_linear_v)
                    w_b_linear_vs.append(w_b_linear_v)
                    
                    _in = self.lora_vit.head.in_features
                    _out = int(melo_info[5])
                    self.num_classes.append(_out)
                    self.fc_loras.append(f.get_tensor(f"fc_{_in}in_{_out}out"))
            
            blk.attn.qkv = _LoRA_qkv_timm_x(
                w_qkv_linear,
                w_a_linear_qs,
                w_b_linear_qs,
                w_a_linear_vs,
                w_b_linear_vs,
                scale_list
            )
        # self.reset_parameters()
        # self.proj_3d = nn.Linear(num_classes * 30, num_classes)
        for file_path in lora_files:
            with safe_open(file_path, framework="pt") as f:
                for key in f.keys():
                    if 'fc_' in key:
                        self.fc_loras.append(f.get_tensor(key))
                        break
        # if num_classes > 0:
        #     self.lora_vit.reset_classifier(num_classes=num_classes)
            # self.lora_vit.head = nn.Linear(
            #     self.dim, num_classes)
    
    def swith_lora(self, idx:int):
        for t_layer_i, blk in enumerate(self.lora_vit.blocks):
            blk.attn.qkv.change_lora(idx)
        self.lora_vit.reset_classifier(num_classes=self.num_classes[idx])
        self.lora_vit.head.weight = Parameter(self.fc_loras[idx])

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)



if __name__ == "__main__":  # Debug
    img = torch.randn(2, 3, 224, 224)
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    lora_vit = LoRA_ViT_timm(vit_model=model, r=4, num_classes=10)
    pred = lora_vit(img)
    print(pred.shape)

    img = torch.randn(2*20, 3, 224, 224)
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    lora_vit = LoRA_ViT_timm(vit_model=model, r=4, num_classes=10)
    pred = lora_vit.forward3D(img)
    print(pred.shape)
