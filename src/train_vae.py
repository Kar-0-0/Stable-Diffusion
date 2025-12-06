import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List
from torchvision.datasets import CocoCaptions
from torchvision import transforms, datasets
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from vae import AutoencoderKL

plt.ion()

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

if __name__ == "__main__":
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
        timesteps=1000,
        base_channels=64,
        batch_size=32,
        num_epochs=15,
        lr=1e-4,
        num_workers=4,
        scaling_factor=0.18215,
        beta=0.00001
    )
    cfg = default_config

    vae = AutoencoderKL(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        latent_channels=cfg.latent_channels,
        block_out_channels=cfg.block_out_channels,
        down_block_types=None,          # or your list later
        up_block_types=None,
        layers_per_block=cfg.layers_per_block,
        norm_num_groups=cfg.norm_num_groups,
        sample_size=cfg.image_size,
        scaling_factor=cfg.scaling_factor,
    )


    transform = make_image_transform(cfg.image_size)
    train_dataset_full = CocoCaptions(
        root=str(data_root),
        annFile=str(captions_json),
        transform=transform,
    )

    # Only using first 1k images because training the full dataset is going to take 49 HOURS!!!
    indices = list(range(40000))
    train_dataset = Subset(train_dataset_full, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=coco_collate_fn,
    )


    viz_fig = None
    viz_ax1 = None
    viz_ax2 = None
    viz_im1 = None
    viz_im2 = None

    def visualize_recon(model):
        global viz_fig, viz_ax1, viz_ax2, viz_im1, viz_im2

        print("Creating Visualization...")

        device = next(model.parameters()).device
        model.eval()

        images, _ = next(iter(train_loader))
        images = images.to(device)

        with torch.no_grad():
            _, _, z = model.encoder(images)
            recon = model.decoder(z)

        orig = images[0].detach().cpu()
        recon0 = recon[0].detach().cpu()

        def to_img(x):
            x = x.permute(1, 2, 0).numpy()
            x = (x * 0.5 + 0.5).clip(0, 1)
            return x

        orig_img = to_img(orig)
        recon_img = to_img(recon0)

        if viz_fig is None:
            # first time: create window and axes
            viz_fig, (viz_ax1, viz_ax2) = plt.subplots(1, 2, figsize=(6, 3))
            viz_im1 = viz_ax1.imshow(orig_img)
            viz_ax1.axis("off")
            viz_ax1.set_title("Original")

            viz_im2 = viz_ax2.imshow(recon_img)
            viz_ax2.axis("off")
            viz_ax2.set_title("Recon")

            plt.tight_layout()
        else:
            # later: just update image data
            viz_im1.set_data(orig_img)
            viz_im2.set_data(recon_img)

        viz_fig.canvas.draw()
        viz_fig.canvas.flush_events()
        plt.savefig('recon')

    # Train
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    vae = vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.lr)

    mse_losses = []
    kl_losses = []
    losses = []

    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    for epoch in range(cfg.num_epochs):
        print(f"Epoch: {epoch}")
        visualize_recon(vae)
        with torch.no_grad():
            images, _ = next(iter(train_loader))
            images = images.to(device)
            mu, _, _ = vae(images)
            print(f"Latent Mean: {mu.mean().item():.3f}, Std: {mu.std().item():.3f}")
        vae.train()
        for i, (image, _) in enumerate(train_loader):
            if i % 10 == 0 or i+1 == len(train_loader):
                print(f"{i}/{len(train_loader)}")
                
            image = image.to(device)
            mu, log_var, recon = vae(image)
            mse_loss = F.mse_loss(recon, image)
            kl_per_sample = -0.5 * torch.sum(1+log_var-mu**2 - torch.exp(log_var), dim=(1, 2, 3))
            kl_loss = torch.mean(kl_per_sample)
            beta = cfg.beta
            loss = mse_loss + beta*kl_loss

            mse_losses.append(mse_loss.item())
            kl_losses.append(kl_loss.item())
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        print(f"MSE Loss: {sum(mse_losses)/len(mse_losses)}\nKL Loss: {sum(kl_losses)/len(kl_losses)}\nTotal Loss: {sum(losses)/len(losses)}")

        mse_losses = []
        kl_losses = []
        losses = []
        if (epoch + 1) % 5 == 0:
            torch.save(vae.state_dict(), f"vae_ckpt_epoch{epoch+1}.pth")