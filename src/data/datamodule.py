import os
import lightning as L
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data.dataset import PairedDataset

class PairedDataModule(L.LightningDataModule):
  def __init__(self, data_dir, batch_size, img_height, img_width):
    super().__init__()
    self.data_dir = data_dir
    self.train_dir = os.path.join(data_dir, 'train')
    self.val_dir = os.path.join(data_dir, 'val')
    self.test_dir = os.path.join(data_dir, 'test')

    self.img_height = img_height
    self.img_width = img_width
    self.batch_size = batch_size
    
    self.transforms_ = [
      transforms.Resize((self.img_height, self.img_width), Image.BICUBIC),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

  def setup(self, stage=None):
    self.train_dataset = PairedDataset(self.train_dir, self.transforms_)
    self.val_dataset = PairedDataset(self.val_dir, self.transforms_)
    self.test_dataset = PairedDataset(self.test_dir, self.transforms_)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2, shuffle=True)
  
  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False)
  
  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False)