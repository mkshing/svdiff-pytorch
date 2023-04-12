import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange



class SVDConv2d(nn.Conv2d):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        scale: float = 1.0,
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        assert type(kernel_size) is int
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vh = torch.linalg.svd(weight_reshaped, full_matrices=False)
        # initialize to 0 for smooth tuning 
        self.delta = nn.Parameter(torch.zeros_like(self.S))
        self.weight.requires_grad = False
        self.done_svd = False
        self.scale = scale
        self.reset_parameters()

    def set_scale(self, scale: float):
        self.scale = scale

    def perform_svd(self):
        # shape
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vh = torch.linalg.svd(weight_reshaped, full_matrices=False)
        self.done_svd = True        
        
    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'delta'):
            nn.init.zeros_(self.delta)

    def forward(self, x: torch.Tensor):
        if not self.done_svd:
            # this happens after loading the state dict 
            self.perform_svd()
        weight_updated = self.U.to(x.device, dtype=x.dtype) @ torch.diag(F.relu(self.S.to(x.device, dtype=x.dtype)+self.scale * self.delta)) @ self.Vh.to(x.device, dtype=x.dtype)
        weight_updated = rearrange(weight_updated, 'co (cin h w) -> co cin h w', cin=self.weight.size(1), h=self.weight.size(2), w=self.weight.size(3))
        return F.conv2d(x, weight_updated, self.bias, self.stride, self.padding, self.dilation, self.groups)
    

class SVDConv1d(nn.Conv1d):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        scale: float = 1.0,
        **kwargs
    ):
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        assert type(kernel_size) is int
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vh = torch.linalg.svd(weight_reshaped, full_matrices=False)
        # initialize to 0 for smooth tuning 
        self.delta = nn.Parameter(torch.zeros_like(self.S))
        self.weight.requires_grad = False
        self.done_svd = False
        self.scale = scale
        self.reset_parameters()

    def set_scale(self, scale: float):
        self.scale = scale

    def perform_svd(self):
        # shape
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vh = torch.linalg.svd(weight_reshaped, full_matrices=False)
        self.done_svd = True        
        
    def reset_parameters(self):
        nn.Conv1d.reset_parameters(self)
        if hasattr(self, 'delta'):
            nn.init.zeros_(self.delta)

    def forward(self, x: torch.Tensor):
        if not self.done_svd:
            # this happens after loading the state dict 
            self.perform_svd()
        weight_updated = self.U.to(x.device, dtype=x.dtype) @ torch.diag(F.relu(self.S.to(x.device, dtype=x.dtype)+self.scale * self.delta)) @ self.Vh.to(x.device, dtype=x.dtype)
        weight_updated = rearrange(weight_updated, 'co (cin h w) -> co cin h w', cin=self.weight.size(1), h=self.weight.size(2), w=self.weight.size(3))
        return F.conv1d(x, weight_updated, self.bias, self.stride, self.padding, self.dilation, self.groups)
    


class SVDLinear(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        scale: float = 1.0,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight, full_matrices=False)        
        # initialize to 0 for smooth tuning 
        self.delta = nn.Parameter(torch.zeros_like(self.S))
        self.weight.requires_grad = False
        self.done_svd = False
        self.scale = scale
        self.reset_parameters()
    
    def set_scale(self, scale: float):
        self.scale = scale

    def perform_svd(self):
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight, full_matrices=False)
        self.done_svd = True    

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'delta'):
            nn.init.zeros_(self.delta)

    def forward(self, x: torch.Tensor):
        if not self.done_svd:
            # this happens after loading the state dict 
            self.perform_svd()
        weight_updated = self.U.to(x.device, dtype=x.dtype) @ torch.diag(F.relu(self.S.to(x.device, dtype=x.dtype)+self.scale * self.delta)) @ self.Vh.to(x.device, dtype=x.dtype)
        return F.linear(x, weight_updated, bias=self.bias)
    

class SVDEmbedding(nn.Embedding):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        scale: float = 1.0,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight, full_matrices=False)
        # initialize to 0 for smooth tuning 
        self.delta = nn.Parameter(torch.zeros_like(self.S))
        self.weight.requires_grad = False
        self.done_svd = False
        self.scale = scale
        self.reset_parameters()
    
    def set_scale(self, scale: float):
        self.scale = scale

    def perform_svd(self):
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight, full_matrices=False)
        self.done_svd = True    

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'delta'):
            nn.init.zeros_(self.delta)

    def forward(self, x: torch.Tensor):
        if not self.done_svd:
            # this happens after loading the state dict 
            self.perform_svd()
        weight_updated = self.U.to(x.device) @ torch.diag(F.relu(self.S.to(x.device)+self.scale * self.delta)) @ self.Vh.to(x.device)
        return F.embedding(x, weight_updated, padding_idx=self.padding_idx, max_norm=self.max_norm, norm_type=self.norm_type, scale_grad_by_freq=self.scale_grad_by_freq, sparse=self.sparse)
    

# 1-D
class SVDLayerNorm(nn.LayerNorm):
    def __init__(
        self, 
        normalized_shape: int, 
        scale: float = 1.0,
        **kwargs
    ):
        nn.LayerNorm.__init__(self, normalized_shape=normalized_shape, **kwargs)
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight.unsqueeze(0), full_matrices=False)
        # initialize to 0 for smooth tuning 
        self.delta = nn.Parameter(torch.zeros_like(self.S))
        self.weight.requires_grad = False
        self.done_svd = False
        self.scale = scale
        self.reset_parameters()
    
    def set_scale(self, scale: float):
        self.scale = scale

    def perform_svd(self):
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight.unsqueeze(0), full_matrices=False)
        self.done_svd = True    

    def reset_parameters(self):
        nn.LayerNorm.reset_parameters(self)
        if hasattr(self, 'delta'):
            nn.init.zeros_(self.delta)

    def forward(self, x: torch.Tensor):
        if not self.done_svd:
            # this happens after loading the state dict 
            self.perform_svd()
        weight_updated = self.U.to(x.device, dtype=x.dtype) @ torch.diag(F.relu(self.S.to(x.device, dtype=x.dtype)+self.scale * self.delta)) @ self.Vh.to(x.device, dtype=x.dtype)
        weight_updated = weight_updated.squeeze(0)
        return F.layer_norm(x, normalized_shape=self.normalized_shape, weight=weight_updated, bias=self.bias, eps=self.eps)
    

class SVDGroupNorm(nn.GroupNorm):
    def __init__(
        self, 
        num_groups: int, 
        num_channels: int,
        scale: float = 1.0,
        **kwargs
    ):
        nn.GroupNorm.__init__(self, num_groups, num_channels, **kwargs)
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight.unsqueeze(0), full_matrices=False)
        # initialize to 0 for smooth tuning 
        self.delta = nn.Parameter(torch.zeros_like(self.S))
        self.weight.requires_grad = False
        self.done_svd = False
        self.scale = scale
        self.reset_parameters()
    
    def set_scale(self, scale: float):
        self.scale = scale

    def perform_svd(self):
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight.unsqueeze(0), full_matrices=False)
        self.done_svd = True    

    def reset_parameters(self):
        nn.GroupNorm.reset_parameters(self)
        if hasattr(self, 'delta'):
            nn.init.zeros_(self.delta)

    def forward(self, x: torch.Tensor):
        if not self.done_svd:
            # this happens after loading the state dict 
            self.perform_svd()
        weight_updated = self.U.to(x.device, dtype=x.dtype) @ torch.diag(F.relu(self.S.to(x.device, dtype=x.dtype)+self.scale * self.delta)) @ self.Vh.to(x.device, dtype=x.dtype)
        weight_updated = weight_updated.squeeze(0)
        return F.group_norm(x, num_groups=self.num_groups, weight=weight_updated, bias=self.bias, eps=self.eps)

