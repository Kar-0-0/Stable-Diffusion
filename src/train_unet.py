import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from typing import List
from torchvision.datasets import CocoCaptions
from torchvision import transforms, datasets
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from vae import AutoencoderKL
import importlib
import unet
importlib.reload(unet)  # force Python to use the latest version
from unet import SDConfig, UNetLatent, TimeEmbedding, ClipTextEncoder
import torchvision
from copy import deepcopy



def make_image_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),                         # [0,1]
        transforms.Normalize(                          # to [-1,1]
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

def coco_collate_fn(batch):
    # batch is a list of (image, captions) pairs
    images = [b[0] for b in batch]          # list of tensors
    captions = [b[1] for b in batch]        # list of lists of strings
    images = torch.stack(images, dim=0)     # (B, 3, H, W)
    return images, captions

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
    block_out_channels=[64, 128, 256, 256],
    layers_per_block=2,
    norm_num_groups=32,
    timesteps=200,
    base_channels=64,
    batch_size=16,
    num_epochs=20,
    lr=1e-4,
    num_workers=4,
    scaling_factor=0.18215,
    beta=0.000001,
    time_dim=256
)

cfg = default_config
def baab(low=1e-4, high=0.02): # beta alpha alpha bar --> baab
    timesteps = cfg.timesteps
    beta = torch.linspace(low, high, timesteps)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    return beta, alpha, alpha_bar

def q_sample(x0, t, noise, alpha_bar):
    # x0.shape = (B, 4, 16, 16)
    # t.shape = (B) type long
    #noise is same shape as x0
    # alpha_bar.shape = (T)

    a_bar_t = alpha_bar[t] # (T) --> (B)
    a_bar_t = a_bar_t.view(-1, 1, 1, 1)

    x_t = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1.0 - a_bar_t) * noise

    return x_t


if __name__ == "__main__":
    cfg = default_config


    transform = make_image_transform(cfg.image_size)
    train_dataset_full = CocoCaptions(
        root=str(data_root),
        annFile=str(captions_json),
        transform=transform,
    )

    #Only using first 1k images because training the full dataset is going to take 49 HOURS!!!
    indices = list(range(20000))
    train_dataset = Subset(train_dataset_full, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=coco_collate_fn,
    )


    vae = AutoencoderKL(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        latent_channels=cfg.latent_channels,
        block_out_channels=[64, 128],
        down_block_types=None,          # or your list later
        up_block_types=None,
        layers_per_block=cfg.layers_per_block,
        norm_num_groups=cfg.norm_num_groups,
        sample_size=cfg.image_size,
        scaling_factor=cfg.scaling_factor,
    )

    

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    vae.load_state_dict(torch.load("vae_ckpt.pth", map_location=device))
    vae.to(device)
    vae.eval()

    t_ember = TimeEmbedding(cfg.time_dim)
    unet = UNetLatent(cfg).to(device)
    unet.to(device=device)
    t_ember.to(device=device)

    ema_unet = deepcopy(unet)
    ema_decay = 0.9999

    def update_ema(ema_model, model, decay):
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    lr = 1e-4
    optimizer = torch.optim.Adam(
        list(unet.parameters()) + list(t_ember.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    print(f"Total params: {sum(p.numel() for p in optimizer.param_groups[0]['params']):,}")

    
    beta, alpha, alpha_bar = baab()
    alpha_bar = alpha_bar.to(device)
    epochs = cfg.num_epochs

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    text_ember = ClipTextEncoder().to(device)
    text_ember.eval()
    for param in text_ember.parameters():
        param.requires_grad = False
    def sample_and_save_image(unet, t_ember, vae, cfg, device, epoch, captions, num_samples=4):
        unet.eval()
        t_ember.eval()

        with torch.no_grad():
            captions = [captions] * num_samples
            text_emb = text_ember(captions)

            latent_shape = (num_samples, cfg.latent_channels, cfg.image_size // 8, cfg.image_size // 8)  # 128/8=16
            x_t = torch.randn(latent_shape, device=device)  # start from noise

            beta, alpha, alpha_bar = baab()
            beta = beta.to(device)
            alpha = alpha.to(device)
            alpha_bar = alpha_bar.to(device)

            for t in reversed(range(cfg.timesteps)):
                t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
                t_emb_batch = t_ember(t_batch)
                eps_pred = unet(x_t, t_emb_batch, text_emb)
                alpha_t = alpha[t]
                alpha_bar_t = alpha_bar[t]
                beta_t = beta[t]

                # DDPM update step
                if t > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = torch.zeros_like(x_t)

                x_t = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_pred) + torch.sqrt(beta_t) * noise

            # Decode latent to image
            x_sampled = vae.decoder(x_t).clamp(-1, 1)  # shape (B, 3, H, W)
            x_sampled = (x_sampled + 1) / 2  # [-1,1] to [0,1]

        # Save images to disk
        grid = torchvision.utils.make_grid(x_sampled.cpu(), nrow=num_samples)
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.title(f'Sampled images at epoch {epoch}\n Prompt: {captions}')
        plt.imshow(grid.permute(1, 2, 0))
        plt.savefig(f'samples_epoch{epoch}.png')
        plt.close()

        unet.train()
        t_ember.train()


     
    for epoch in range(epochs):
        losses = []
        for i, (image, labels) in enumerate(train_loader):
            x = image.to(device)
            with torch.no_grad():
                mu, log_var, _ = vae(x)          # x: images
                sigma = torch.exp(0.5 * log_var)
                z = (mu + sigma * torch.randn_like(sigma))

            t = torch.randint(0, cfg.timesteps, (x.size(0),), device=device)
            if i % 100 == 0 or i + 1 == len(train_loader):
                 print(f"{i}/{len(train_loader)} | t_range: [{t.min().item()}-{t.max().item()}] mean: {t.float().mean().item():.1f}")
            eps = torch.randn_like(z)
            x_t = q_sample(z, t, eps, alpha_bar)
            t_emb = t_ember(t)
            label_flat = [lab[0] for lab in labels]
            with torch.no_grad():
                text_embs = text_ember(label_flat)
            eps_pred = unet(x_t, t_emb, text_embs)
            loss = F.mse_loss(eps_pred, eps)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 1.0)
            optimizer.step()
            update_ema(ema_unet, unet, ema_decay)

        scheduler.step()
        print(f"Epoch {epoch}\nLoss: {sum(losses)/len(losses)}")
        if epoch > 0 and (epoch + 1) % 5 == 0:
            e = epoch + 1     # human-readable epoch number
            torch.save(unet.state_dict(), f"unet_ckpt{e}.pth")
            torch.save(ema_unet.state_dict(), f"ema_unet_ckpt{e}.pth")
            torch.save(t_ember.state_dict(), f"t_ember_ckpt{e}.pth")
        
        ix = torch.randint(0, len(train_loader)-1, (1,)).item()
        _, labels_batch = next(iter(DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=coco_collate_fn)))
        caption = labels_batch[0][0] 
        sample_and_save_image(ema_unet, t_ember, vae, cfg, device, epoch, "a dog on the grass", 4)

    # Save for Sampling
    cfg_dict = asdict(cfg)

    import json
    with open("config.json", 'w') as f:
        json.dump(cfg_dict, f)