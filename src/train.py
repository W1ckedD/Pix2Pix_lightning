import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.profilers import PyTorchProfiler

from src.models.pix2pix import Pix2Pix
from src.data.datamodule import PairedDataModule
from src.config import config

def main():
  # Data
  dm = PairedDataModule(
    data_dir=config.DATA_PATH,
    batch_size=config.BATCH_SIZE,
    img_height=config.IMG_HEIGHT,
    img_width=config.IMG_WIDTH
  )

  # Model
  model = Pix2Pix(
    img_height=config.IMG_HEIGHT,
    img_width=config.IMG_WIDTH,
    b1=config.B1,
    b2=config.B2,
    lr=config.LR,
    lambda_pixel=config.LAMBDA_PIXEL,
    sample_interval=config.SAMPLE_INTERVAL,
    use_perceptual=config.USE_PERCEPTUAL
  )

  # Loggers
  logger = TensorBoardLogger(config.LOG_DIR, name=f'pix2pix-v{config.VERSION}')

  # Callbacks
  checkpoint_callback = ModelCheckpoint(
    dirpath=f'checkpoints-v{config.VERSION}/',
    filename='pix2pix-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    monitor='val_loss_G',
    mode='min'
  )

  early_stopping = EarlyStopping(
    monitor='val_loss_G',
    patience=5,
    mode='min'
  )

  # Profiler
  profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{config.LOG_DIR}/profiler"),
    trace_memory=True,
    profile_memory=True,
    schedule=torch.profiler.schedule(skip_first=0, wait=0, warmup=0, active=10)
  )

  trainer = L.Trainer(
    min_epochs=1,
    max_epochs=config.N_EPOCHS,
    devices=config.DEVICES,
    accelerator=config.ACCELERATOR,
    strategy=config.STRATEGY,
    logger=logger,
    callbacks=[checkpoint_callback],
    # profiler=profiler
  )
  trainer.fit(model, dm)
  trainer.validate(model, dm)
  trainer.test(model, dm)

if __name__ == '__main__':
  main()