import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import torch.nn as nn

# Custom weights initialization called on netG and netD with mean = 0 and std = 0.02
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Plot loss curve for G and D
def plot_loss_curves(G_losses, D_losses):
    """Saves a plot of generator and discriminator loss curves."""
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("results/loss_curves.png")
    plt.close()

def plot_real_vs_fake(device, dataloader, img_list):
    """Saves a comparison of real and fake images."""
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    
    plt.savefig("results/real_vs_fake.png")
    plt.close()