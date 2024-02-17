import torch
import torch.nn as nn
import lightning as L
from src.networks.UNet import GeneratorUNet
from src.networks.PatchGAN import Discriminator

class Pix2Pix(L.LightningModule):
  def __init__(self):
    super(Pix2Pix, self).__init__()
    self.automatic_optimization = False

    self.generator = GeneratorUNet(in_channels=3, out_channels=3)
    self.discriminator = Discriminator(in_channels=3)


    # TODO: Define the loss function
    # TODO: Initialize metrics
    # TODO: add visualization