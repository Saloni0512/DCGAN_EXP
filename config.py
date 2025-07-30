import torch

# device agnostic setup
if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Training hyperparameters
LEARNING_RATE_D = 0.0002
LEARNING_RATE_G = 0.0002
BETA1 = 0.5
BATCH_SIZE = 128
NUM_EPOCHS = 15
NUM_WORKERS = 2

# Establish Convention for real and fake images
REAL_LABEL = 0.9  # Using label smoothing for better images
FAKE_LABEL = 0

# Model hyperparameters
IMAGE_SIZE = 32
NC = 1 # no of color channels set to 1 since FashionMNIST images are grayscale
NZ = 100  # Size of the latent z vector
NGF = 64  # Generator feature map size
NDF = 64  # Discriminator feature map size
NGPU = 1 # Set to 1 as MPS doesn't use DataParallel

# set seed for reproducibility
MANUAL_SEED = 999