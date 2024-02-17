# Training params
LR = 0.0001
B1 = 0.5
B2 = 0.999
BATCH_SIZE = 8
N_EPOCHS = 100

# Model params
IMG_HEIGHT = 512
IMG_WIDTH = 512
USE_PERCEPTUAL = True
LAMBDA_PIXEL = 100

# Data params
DATA_PATH = 'sample_dataset/'


# Log params
SAMPLE_INTERVAL = 50

# Hardware params
DEVICES = 1
ACCELERATOR = 'cpu'
STRATEGY = 'auto'
