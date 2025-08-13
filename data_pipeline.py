import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
import config 

## Data Loading
# Image transformation pipeline
transform = transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE, InterpolationMode.BILINEAR), # 28x28 -> 32x32
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
])

# Load dataset
dataset = datasets.FashionMNIST(root="data",
                                train=True,
                                download=True,
                                transform=transform)
# Create dataloader   
dataloader = DataLoader(dataset=dataset,
                        batch_size=config.BATCH_SIZE,
                        num_workers=config.NUM_WORKERS,
                        shuffle=True
) 