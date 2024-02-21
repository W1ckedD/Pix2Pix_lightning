# Version
VERSION = 2

# Training params
LR = 0.0001
B1 = 0.5
B2 = 0.999
BATCH_SIZE = 16
N_EPOCHS = 100

# Model params
IMG_HEIGHT = 512
IMG_WIDTH = 512
USE_PERCEPTUAL = True
LAMBDA_PIXEL = 1

# Data params
DATA_PATH = 'data_2360/'


# Log params
SAMPLE_INTERVAL = 20
LOG_DIR = 'logs'

# Hardware params
DEVICES = 1
ACCELERATOR = 'cuda'
STRATEGY = 'auto'
