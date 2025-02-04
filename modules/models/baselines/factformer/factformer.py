import torch
import torch.nn as nn
from einops import rearrange
from modules.models.baselines.factformer.factorization_module import FABlock3D
from modules.models.baselines.factformer.positional_encoding_module import GaussianFourierFeatureTransform

class FactorizedTransformer(nn.Module):
    def __init__(self,
                 dim,
                 dim_head,
                 heads,
                 dim_out,
                 dim_in,
                 depth,
                 **kwargs):
        super().__init__()
        self.to_in = nn.Linear(dim_in, dim)
        self.to_out = nn.Linear(dim, dim_out)
        self.layers = nn.ModuleList([])
        for _ in range(depth):

            layer = nn.ModuleList([])
            layer.append(nn.Sequential(
                GaussianFourierFeatureTransform(3, dim // 2, 1),
                nn.Linear(dim, dim)
            ))
            layer.append(FABlock3D(dim, dim_head, dim, heads, dim, use_rope=True, **kwargs))
            self.layers.append(layer)

    def forward(self, u, pos_lst):
        b, nx, ny, nz, c = u.shape  # just want to make sure its shape
        u = self.to_in(u)
        nx, ny, nz = pos_lst[0].shape[0], pos_lst[1].shape[0], pos_lst[2].shape[0]
        pos = torch.stack(torch.meshgrid([pos_lst[0].squeeze(-1),
                                          pos_lst[1].squeeze(-1),
                                          pos_lst[2].squeeze(-1)]
                                         ), dim=-1).reshape(-1, 3)

        for l, (pos_enc, attn_layer) in enumerate(self.layers):
            u += rearrange(pos_enc(pos), '1 (nx ny nz) c -> 1 nx ny nz c', nx=nx, ny=ny, nz=nz)
            u = attn_layer(u, pos_lst) + u
        
        u = self.to_out(u)
        return u