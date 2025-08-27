import torch.nn as nn
import config
import torch

class Generator(nn.Module):
  """The generator class module in dcgan for generating fake images."""
  def __init__(self, ngpu):
    super(Generator, self).__init__()
    self.ngpu = ngpu
    
    # Takes latent space vector NZ as input
    self.block1 = nn.Sequential(
        nn.ConvTranspose2d(config.NZ, config.NGF * 4, 4, 1, 0, bias=False),
        nn.BatchNorm2d(config.NGF * 4),
        nn.ReLU(True),
    )    
    # Upsamples to 8x8
    self.block2 = nn.Sequential(
        nn.ConvTranspose2d(config.NGF * 4, config.NGF * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(config.NGF * 2),
        nn.ReLU(True),
    )    
    # Upsamples to 16x16
    self.block3 = nn.Sequential(    
        nn.ConvTranspose2d(config.NGF * 2, config.NGF, 4, 2, 1, bias=False),
        nn.BatchNorm2d(config.NGF),
        nn.ReLU(True),
    )    
    # Upsamples to 32x32
    self.block4 = nn.Sequential(    
        nn.ConvTranspose2d(config.NGF, config.NC, 4, 2, 1, bias=False),
        nn.Tanh()
    )
  
  def forward(self, input):
    # Check intermediate tensor shapes after each block
    x = self.block1(input)
    #print(f"After Block 1 (4x4): \t\t{x.shape}")
    
    x = self.block2(x)
    #print(f"After Block 2 (8x8): \t\t{x.shape}")
    
    x = self.block3(x)
    #print(f"After Block 3 (16x16): \t\t{x.shape}")
    
    x = self.block4(x)
    #print(f"After Block 4 (Output 32x32): \t{x.shape}")
    
    return x
    
    


class Discriminator(nn.Module):
  """The discrimator class module in dcgan for classifying fake and real images."""
  def __init__(self, ngpu):
    super(Discriminator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
        nn.Conv2d(config.NC, config.NDF, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
       
        nn.Conv2d(config.NDF, config.NDF * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(config.NDF * 2),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(config.NDF * 2, config.NDF * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(config.NDF * 4),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(config.NDF * 4, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()
    )
  
  def forward(self, input):
    return self.main(input)
  


# --- Model tests ---
# if __name__ == '__main__':
# --- Test the Generator ---
#   gen = Generator(ngpu=config.NGPU).to(config.DEVICE)
#   print(f"Testing tensor shape values in Generator...")
 # Create a random latent vector
 #  noise = torch.randn(config.BATCH_SIZE, config.NZ, 1, 1, device=config.DEVICE)
 #  fake_img = gen(noise)
 #  print("-------------------------\n")


