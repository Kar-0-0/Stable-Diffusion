import torch 
import torch.nn as nn 
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
import math
from dataclasses import dataclass
from typing import List
from pathlib import Path


@dataclass
class SDConfig:
    # data
    data_root: str          # path to COCO images
    captions_json: str      # path to captions json (train2017)
    image_size: int         # 128 for your project

    # model – VAE
    in_channels: int        # 3 (RGB)
    out_channels: int       # 3
    latent_channels: int    # e.g. 4
    block_out_channels: List[int]  # e.g. [64, 128, 256]
    layers_per_block: int   # e.g. 2
    norm_num_groups: int    # e.g. 32

    # diffusion / U-Net (later)
    timesteps: int          # e.g. 1000
    base_channels: int      # U-Net width (e.g. 64)

    # training
    batch_size: int
    num_epochs: int
    lr: float
    num_workers: int
    beta: float

    # VAE scaling
    scaling_factor: float

    # U-Net
    time_dim: int



PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_root = PROJECT_ROOT / "data" / "coco" / "train2017"
captions_json = PROJECT_ROOT / "data" / "coco" / "annotations" / "captions_train2017.json"

default_config = SDConfig(
    data_root=str(data_root),
    captions_json=str(captions_json),
    image_size=128,
    in_channels=3,
    out_channels=3,
    latent_channels=4,
    block_out_channels=[64, 128, 256],
    layers_per_block=2,
    norm_num_groups=32,
    timesteps=1000,
    base_channels=64,
    batch_size=16,
    num_epochs=15,
    lr=1e-4,
    num_workers=4,
    scaling_factor=0.18215,
    beta=0.000001,
    time_dim=256
)
cfg = default_config



class TimeResNet2D(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            norm_num_groups,
            time_dim
    ):
        super().__init__()
        self.gn1 = nn.GroupNorm(norm_num_groups, in_channels)
        self.silu1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.gn2 = nn.GroupNorm(norm_num_groups, out_channels)
        self.silu2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.t_proj = nn.Linear(time_dim, out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.conv1(self.silu1(self.gn1(x)))
        t_emb = self.t_proj(t_emb)
        t_emb = t_emb.unsqueeze(dim=-1).unsqueeze(dim=-1)
        h = h + t_emb 
        h = self.conv2(self.silu2(self.gn2(h)))

        return self.skip(x) + h


class DownBlock2D(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels,
            norm_num_groups,
            time_dim,
            layers_per_block,
            add_downsample=True
    ):
        super().__init__()
        self.resnets = nn.ModuleList()

        ch_in = in_channels
        for _ in range(layers_per_block):
            self.resnets.append(
                TimeResNet2D(
                    ch_in,
                    out_channels,
                    norm_num_groups,
                    time_dim
                )
            )
            ch_in = out_channels

        self.add_downsample = add_downsample
        if add_downsample:
            self.downsample = nn.Conv2d(
                out_channels,
                out_channels, 
                kernel_size=3,
                stride=2,
                padding=1
            )
    
    def forward(self, x, t_emb):
        x_pre = x
        for resnet in self.resnets:
            x_pre = resnet(x_pre, t_emb)

        if self.add_downsample:
            x_down = self.downsample(x_pre)
        else:
            x_down = None

        return x_pre, x_down


class UnetEncoder(nn.Module):
    def __init__(
            self,
            in_channels,
            block_out_channels,
            norm_num_groups, 
            layers_per_block,
            time_dim
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()

        seq_len = len(block_out_channels)
        for i in range(seq_len):
            in_channels = block_out_channels[i-1] if i > 0 else block_out_channels[0]
            out_channels = block_out_channels[i]
            add_downsample = (i < seq_len-1)
            self.down_blocks.append(
                DownBlock2D(
                    in_channels,
                    out_channels,
                    norm_num_groups,
                    time_dim,
                    layers_per_block,
                    add_downsample
                )
            )
    
    def forward(self, x, t_emb):
        skips = []

        x = self.conv_in(x)

        for down_block in self.down_blocks:
            x_pre, x_down = down_block(x, t_emb)
            skips.append(x_pre)
            x = x_down if x_down is not None else x_pre
        
        return x, skips


class SelfAttention(nn.Module):
    def __init__(
            self, 
            channels, # latent channels
            time_dim,            
    ):
        super().__init__()

        self.c_attn = nn.Linear(channels, channels*3)
        self.c_proj = nn.Linear(channels, channels)
        self.t_proj = nn.Linear(time_dim, channels)
        self.n_heads = 4 
 
    
    def forward(self, x, t_emb):
        # x ---> (B, C, H, W)
        B, C, H, W = x.shape
        T = H*W
        x = x.view(B, C, T).permute(0, 2, 1)
        t_emb = self.t_proj(t_emb).unsqueeze(1) # B, 1, C
        x = x + t_emb
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2) # B, H*W, C
        q = q.view(B, T, self.n_heads, C//self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_heads, C//self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C//self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        attn = (q @ k.transpose(-2, -1)) * 1/math.sqrt(k.size(-1)) # (B, nh, T, T)
        attn = F.softmax(attn, dim=-1)


        out = attn @ v # (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        out = out.transpose(1, 2).contiguous().view(B, C, H, W)

        return out

class FeedForward(nn.Module):
    def __init__(self, channels): # channels = latent channels
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, 4*channels),
            nn.SiLU(),
            nn.Linear(4*channels, channels)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        T = H*W
        x = x.view(B, C, T).transpose(1, 2)
        x = self.net(x)
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)

        return x


class TransformerBlock(nn.Module):
    def __init__(
            self,
            channels,
            time_dim,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(channels)
        self.sa = SelfAttention(channels, time_dim)
        self.ffwd = FeedForward(channels)
        self.ln2 = nn.LayerNorm(channels)
    
    def forward(self, x, t_emb):
        B, C, H, W = x.shape
        T = H*W
        h = x.view(B, C, T).permute(0, 2, 1) # (B, T, C)
        h = self.ln1(h)
        h = h.permute(0, 2, 1).contiguous().view(B, C, H, W)
        x = x + self.sa(h, t_emb)
        h = x.view(B, C, T).permute(0, 2, 1) # (B, T, C)
        h = self.ln2(h)
        h = h.permute(0, 2, 1).contiguous().view(B, C, H, W)
        x = x + self.ffwd(h)

        return x


class ClipTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_embeddings = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    
    def forward(self, text):
        # text ---> (B)
        tokens = self.tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )
        device = next(self.text_embeddings.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        outputs = self.text_embeddings(**tokens)
        tok_embs = outputs.last_hidden_state  # (B, 77, 512)

        return tok_embs


class CrossAttention(nn.Module):
    def __init__(self, channels, text_dim=512):
        super().__init__()
        self.n_heads = 4
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(text_dim, channels*2)
        self.c_proj = nn.Linear(channels, channels)
    
    def forward(self, x, text_embs):
        # text_embs.shape --> (B, 77, text_dim)
        B, C, H, W = x.shape
        T = H * W
        L = text_embs.size(1)
        x = x.view(B, C, T)
        x = x.permute(0, 2, 1)
        q = self.q(x).view(B, T, self.n_heads, C//self.n_heads).permute(0, 2, 1, 3) # (B, nh, T, hs)
        kv = self.kv(text_embs) # (B, L, td) ---> (B, L, C)
        k, v = kv.split(C, dim=2)
        k = k.view(B, L, self.n_heads, C//self.n_heads).permute(0, 2, 1, 3) # (B, nh, L, hs)
        v = v.view(B, L, self.n_heads, C//self.n_heads).permute(0, 2, 1, 3) # (B, nh, L, hs)
        attn = (q @ k.transpose(-2, -1)) * 1/math.sqrt(k.size(-1)) # (B, nh, T, hs) @ (B, nh, hs, L) ----> (B, nh, T, L)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v # (B, nh, T, L) @ (B, nh, L, hs) ---> (B, nh, T, hs) 
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)

        return out 
    

class MidBlock(nn.Module):
    def __init__(
            self, 
            channels,
            norm_num_groups,
            time_dim
    ):
        super().__init__()

        self.resnet1 = TimeResNet2D(
            channels,
            channels,
            norm_num_groups,
            time_dim
        )

        self.transformer = TransformerBlock(channels, time_dim)
        
        self.clip = ClipTextEncoder()
        self.cross_attention = CrossAttention(channels)

        self.resnet2 = TimeResNet2D(
            channels,
            channels,
            norm_num_groups,
            time_dim
        )
    
    def forward(self, x, t_emb, text_emb):
        x = self.resnet1(x, t_emb)
        x = self.transformer(x, t_emb)
        x = x + self.cross_attention(x, text_emb)
        x = self.resnet2(x, t_emb)

        return x


class UpBlock2D(nn.Module):
    def __init__(
            self, 
            in_channels,
            skip_channels,
            out_channels,
            norm_num_groups, 
            time_dim,
            layers_per_block,
            add_upsample=True
    ):
        super().__init__()

        self.add_upsample = add_upsample
        if add_upsample:
            self.upsample = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        cat_channels = in_channels + skip_channels

        self.resnets = nn.ModuleList()

        in_channels = cat_channels
        for _ in range(layers_per_block):
            self.resnets.append(
                TimeResNet2D(
                    in_channels,
                    out_channels,
                    norm_num_groups,
                    time_dim
                )
            )
            in_channels = out_channels
        
    def forward(self, x, skip, t_emb):
        x = torch.cat([x, skip], dim=1)

        for resnet in self.resnets:
            x = resnet(x, t_emb)

        if self.add_upsample:
            x = self.upsample(x)


        
        return x


class UnetDecoder(nn.Module):
    def __init__(
            self, 
            norm_num_groups,
            layers_per_block,
            time_dim
    ):
        super().__init__()

        self.up_block_low = UpBlock2D(
            in_channels=256,
            skip_channels=256,
            out_channels=256,
            norm_num_groups=norm_num_groups,
            time_dim=time_dim,
            layers_per_block=layers_per_block,
            add_upsample=True,
        )

        self.up_block_mid1 = UpBlock2D(
            in_channels=256,
            skip_channels=256,
            out_channels=256,
            norm_num_groups=norm_num_groups,
            time_dim=time_dim,
            layers_per_block=layers_per_block,
            add_upsample=True,
        )

        self.up_block_mid2 = UpBlock2D(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
            norm_num_groups=norm_num_groups,
            time_dim=time_dim,
            layers_per_block=layers_per_block,
            add_upsample=True,
        )

        self.up_block_high = UpBlock2D(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
            norm_num_groups=norm_num_groups,
            time_dim=time_dim,
            layers_per_block=layers_per_block,
            add_upsample=False,
        )

        self.up_blocks = nn.ModuleList([self.up_block_low, self.up_block_mid1, self.up_block_mid2, self.up_block_high])
    
    def forward(self, x, skips, t_emb):
        reversed_skips = skips[::-1]

        for skip, up_block in zip(reversed_skips, self.up_blocks):
            x = up_block(x, skip, t_emb)
        
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim):
        super().__init__()

        self.time_dim = time_dim

        self.net = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

    def forward(self, t):
        d = self.time_dim
        half_dim = d // 2
        device = t.device
        freq = torch.exp(torch.arange(0, half_dim, device=device) * -(torch.log(torch.tensor(10000.0, device=device))/half_dim))
        freq = freq.view(1, -1)
        t = t.view(t.shape[0], 1)
        args = t.float().to(device) * freq
        sin_part = torch.sin(args)
        cos_part = torch.cos(args)

        out = torch.cat([sin_part, cos_part], dim=-1)

        out = self.net(out)

        return out


class UNetLatent(nn.Module):
    def __init__(self,cfg):  
        super().__init__()

        self.in_channels = cfg.latent_channels
        self.out_channels = cfg.latent_channels
        self.block_out_channels = cfg.block_out_channels
        self.time_dim = cfg.time_dim
        self.norm_num_groups = cfg.norm_num_groups
        self.layers_per_block = cfg.layers_per_block

        self.encoder = UnetEncoder(
            self.in_channels,
            self.block_out_channels,
            self.norm_num_groups,
            self.layers_per_block,
            self.time_dim
        )

        self.mid_block = MidBlock(
            self.block_out_channels[-1],
            self.norm_num_groups,
            self.time_dim
        )
        
        self.decoder = UnetDecoder(
            self.norm_num_groups,
            self.layers_per_block,
            self.time_dim
        )

        self.conv_out = nn.Conv2d(64, self.out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t_emb, text_emb):
        x, skips = self.encoder(x, t_emb)
        x = self.mid_block(x, t_emb, text_emb)
        x = self.decoder(x, skips, t_emb)
        x = self.conv_out(x)

        return x
