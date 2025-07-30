import torch.nn as nn
import config

class Generator(nn.Module):
  """The generator class module in dcgan for generating fake images."""
  def __init__(self, ngpu):
    super(Generator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
        # Takes latent space vector NZ as input
        nn.ConvTranspose2d(config.NZ, config.NGF * 4, 4, 1, 0, bias=False),
        nn.BatchNorm2d(config.NGF * 4),
        nn.ReLU(True),
        # Upsamples to 8x8
        nn.ConvTranspose2d(config.NGF * 4, config.NGF * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(config.NGF * 2),
        nn.ReLU(True),
        # Upsamples to 16x16
        nn.ConvTranspose2d(config.NGF * 2, config.NGF, 4, 2, 1, bias=False),
        nn.BatchNorm2d(config.NGF),
        nn.ReLU(True),
        # Upsamples to 32x32
        nn.ConvTranspose2d(config.NGF, config.NC, 4, 2, 1, bias=False),
        nn.Tanh()
    )
  
  def forward(self, input):
    return self.main(input)


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