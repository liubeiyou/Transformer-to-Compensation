"""
Thank you so much
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
Thank you so much
"""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    translate tensor 240x1x3 to 240x256x32
    """

    def __init__(self, in_c=3, out_c=32, embed_dim=256):
        super(PatchEmbed, self).__init__()
        self.up = nn.Linear(in_features=in_c, out_features=embed_dim, bias=False)
        self.Linear = nn.Linear(in_features=1, out_features=out_c, bias=False)
        self.act = nn.GELU()
        self.num_patches = out_c
        self.embed_dim = embed_dim

    def forward(self, x):  # embedded layer    [B,1,3] -> [B,32,256]
        x = self.up(x)
        x = self.act(x).permute(0, 2, 1)
        x = self.Linear(x).permute(0, 2, 1)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # dim of the input token
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None, ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # 得到结果的每一行进行softmax

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 out_c,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm1d):
        super(Block, self).__init__()
        self.norm1 = norm_layer(out_c)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(out_c)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, in_c=3, out_c=32, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, embed_layer=PatchEmbed, norm_layer=nn.BatchNorm1d,
                 act_layer=None):
        """
        Args:
            in_c (int): number of input channels
            out_c (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(Transformer, self).__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(in_c=in_c, out_c=out_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, out_c=out_c, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(out_c)

        # Predict Head
        self.head = nn.Sequential(nn.Linear(8192, 1000), nn.GELU(),
                                  nn.Linear(1000, 1))

    def forward_features(self, x):
        x = self.patch_embed(x)  # [B,1,3] -> [B, out_c, embed_dim]
        x = x + self.pos_embed  # position embedding
        x = self.blocks(x)  # transformer encoder
        x = self.norm(x)  # Batch_norm
        return x.flatten(1)  # [B, out_c, embed_dim] -> [B, out_c * embed_dim]

    def forward(self, x):
        x = self.forward_features(x)  # Get the characteristic tensor from input[B,1,3]->output[B,out_c*embed_dim]
        x = self.head(x)  # predict head [B,out_c*embed_dim]->[B,1]
        return x.squeeze(1)  # To calculate loss [B,1]->[B]


def bem_au(pred, labels):  # calculate the accuracy 4%
    b = labels.shape
    pred = pred.detach()
    labels = labels.detach()
    x = abs(pred - labels)
    x1 = labels * 0.04
    x = torch.le(x, x1)
    x = x.sum(dim=-1).item()
    x = x / b[0]
    return x


def bem_au_10(pred, labels):  # calculate the accuracy 1%
    b = labels.shape
    pred = pred.detach()
    labels = labels.detach()
    x = abs(pred - labels)
    x1 = labels * 0.01
    x = torch.le(x, x1)
    x = x.sum(dim=-1).item()
    x = x / b[0]
    return x
