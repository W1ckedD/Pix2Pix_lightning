import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PairedDataset(Dataset):
  def __init__(self, data_dir, transforms_=None):
    super(PairedDataset, self).__init__()
    self.transform = transforms.Compose(transforms_)
    self.files_A_dir = os.path.join(data_dir, 'rgb')
    self.files_B_dir = os.path.join(data_dir, 'tactile')
    self.files_A = sorted(os.listdir(self.files_A_dir))
    self.files_B = sorted(os.listdir(self.files_B_dir))

  def __getitem__(self, index):
    tactile = Image.open(os.path.join(self.files_B_dir, self.files_B[index])).convert('RGB')
    rgb = Image.open(os.path.join(self.files_A_dir, self.files_A[index])).convert('RGB')

    tactile = self.transform(tactile)
    rgb = self.transform(rgb)

    return {'A': tactile, 'B': rgb}
  
  def __len__(self):
    return len(self.files_A)