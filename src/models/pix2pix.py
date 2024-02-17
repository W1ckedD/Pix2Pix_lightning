import os
import torch
import torch.nn as nn
import lightning as L
import numpy as np
from torchvision.utils import make_grid
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.networks.UNet import GeneratorUNet
from src.networks.PatchGAN import Discriminator
from src.utils.perceptual_loss import VGGPerceptualLoss

class Pix2Pix(L.LightningModule):
  def __init__(self, img_height=512, img_width=512, b1=0.5, b2=0.999, lr=0.0001, lambda_pixel=100, sample_interval=50, use_perceptual=True):
    super(Pix2Pix, self).__init__()

    self.automatic_optimization = False

    self.img_height = img_height
    self.img_width = img_width
    self.b1 = b1
    self.b2 = b2
    self.lr = lr
    self.lambda_pixel = lambda_pixel
    self.sample_interval = sample_interval
    self.use_perceptual = use_perceptual

    # Initialize generator and discriminator
    self.generator = GeneratorUNet(in_channels=3, out_channels=3)
    self.discriminator = Discriminator(in_channels_a=3, in_channels_b=3)

    # Loss functions
    self.criterion_GAN = nn.BCEWithLogitsLoss()
    self.criterion_pixelwise = nn.L1Loss()

    if self.use_perceptual:
      self.criterion_perceptual = VGGPerceptualLoss()

    self.patch = (1, self.img_height // 2 ** 4, self.img_width // 2 ** 4)

    self.init_metrics()
      
  
  
  def forward(self, x):
    return self.generator(x)
  
  def training_step(self, batch, batch_idx):
    optim_G, optim_D = self.optimizers()
    real_A, real_B = batch['A'], batch['B']


    # Train Generator
    optim_G.zero_grad()

    valid = torch.autograd.Variable(torch.Tensor(np.ones((real_A.size(0), *self.patch))), requires_grad=False).to(self.device)
    fake = torch.autograd.Variable(torch.Tensor(np.zeros((real_B.size(0), *self.patch))), requires_grad=False).to(self.device)

    fake_B = self.generator(real_A)
    pred_fake = self.discriminator(fake_B, real_A)

    loss_GAN = self.criterion_GAN(pred_fake, valid)
    loss_pixel = self.criterion_pixelwise(fake_B, real_B)

    loss_G = loss_GAN + self.lambda_pixel * loss_pixel

    if self.use_perceptual:
      loss_perceptual = self.criterion_perceptual(fake_B, real_B)
      loss_G += loss_perceptual

    loss_G.backward()
    optim_G.step()

    # Train Discriminator
    optim_D.zero_grad()
    pred_real = self.discriminator(real_B, real_A)
    loss_real = self.criterion_GAN(pred_real, valid)

    pred_fake = self.discriminator(fake_B.detach(), real_A)
    loss_fake = self.criterion_GAN(pred_fake, fake)

    loss_D = 0.5 * (loss_real + loss_fake)
    loss_D.backward()
    optim_D.step()

    self.log_dict({
      'train_loss_G': loss_G.item(),
      'train_loss_pixel': loss_pixel.item(),
      'train_loss_GAN': loss_GAN.item(),
      'train_loss_percetual': loss_perceptual.item() if self.use_perceptual else 'N/A',
      'train_loss_D': loss_D.item()
    }, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    if batch_idx % self.sample_interval == 0:
      self.sample_images(real_A, real_B, fake_B, batch_idx)

  def validation_step(self, batch, batch_idx):
    real_A, real_B = batch['A'], batch['B']
    with torch.no_grad():
      valid = torch.autograd.Variable(torch.Tensor(np.ones((real_A.size(0), *self.patch))), requires_grad=False).to(self.device)
      fake_B = self.forward(real_A)
      pred_fake = self.discriminator(fake_B, real_A)

      loss_GAN = self.criterion_GAN(pred_fake, valid)
      loss_pixel = self.criterion_pixelwise(fake_B, real_B)

      if self.use_perceptual:
        loss_perceptual = self.criterion_perceptual(fake_B, real_B)

      self.log_dict({
        'val_loss_G': loss_GAN.item() + self.lambda_pixel * loss_pixel.item() + loss_perceptual.item(),
        'val_loss_pixel': loss_pixel.item(),
        'val_loss_GAN': loss_GAN.item(),
        'val_loss_percetual': loss_perceptual.item() if self.use_perceptual else 'N/A'
      }, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
      

  def test_step(self, batch, batch_idx):
    real_A, real_B = batch['A'], batch['B']
    with torch.no_grad():
      valid = torch.autograd.Variable(torch.Tensor(np.ones((real_A.size(0), *self.patch))), requires_grad=False).to(self.device)
      fake_B = self.forward(real_A)
      pred_fake = self.discriminator(fake_B, real_A)

      loss_GAN = self.criterion_GAN(pred_fake, valid)
      loss_pixel = self.criterion_pixelwise(fake_B, real_B)

      if self.use_perceptual:
        loss_perceptual = self.criterion_perceptual(fake_B, real_B)
        
      self.log_dict({
        'test_loss_G': loss_GAN.item() + self.lambda_pixel * loss_pixel.item() + loss_perceptual.item(),
        'test_loss_pixel': loss_pixel.item(),
        'test_loss_GAN': loss_GAN.item(),
        'test_loss_percetual': loss_perceptual.item() if self.use_perceptual else 'N/A'
      }, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

      self.update_metrics(real_B, fake_B)

  def predict_step(self, batch, batch_idx, dataloader_idx):
    pass

  def on_test_epoch_end(self):
    metrics = self.compute_metrics()
    self.log_dict(metrics, sync_dist=True)

  def configure_optimizers(self):
    optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
    optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
    return optimizer_G, optimizer_D

  def sample_images(self, real_A, real_B, fake_B, batch_idx):
    grid = make_grid([*real_A, *real_B, *fake_B], nrow=real_A.size(0))
    self.logger.experiment.add_image(f'samples_{batch_idx}', grid, self.global_step)

  def init_metrics(self):
    self.fid = FrechetInceptionDistance(feature=64, normalize=True)
    self.psnr = PeakSignalNoiseRatio()
    self.ssim = StructuralSimilarityIndexMeasure()
    self.lpips = LearnedPerceptualImagePatchSimilarity('alex')

    self.psnr_values = []
    self.ssim_values = []
    self.lpips_values = []

  def update_metrics(self, real_B, fake_B):
    # FID
    self.fid.update(real_B, real=True)
    self.fid.update(fake_B, real=False)

    # PSNR
    self.psnr_values.append(self.psnr(fake_B, real_B))

    # SSIM
    self.ssim_values.append(self.ssim(fake_B, real_B))

    # LPIPS
    self.lpips_values.append(self.lpips(fake_B, real_B))

  def compute_metrics(self):
    metrics = {
      'fid': self.fid.compute(),
      'psnr': torch.mean(torch.stack(self.psnr_values)).item(),
      'ssim': torch.mean(torch.stack(self.ssim_values)).item(),
      'lpips': torch.mean(torch.stack(self.lpips_values)).item()
    }

    return metrics

    