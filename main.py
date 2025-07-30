import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import os
import random
from timeit import default_timer as timer

# Imports from other files
import config
from model import Generator, Discriminator
from utils import weights_init, plot_loss_curves, plot_real_vs_fake 

def train():

    # Output directory for storing results
    os.makedirs("results", exist_ok=True)

    # Set random seed for reproducibility
    random.seed(config.MANUAL_SEED)
    torch.manual_seed(config.MANUAL_SEED)

    ## Data Loading
    # Image transformation pipeline
    transform = transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE, InterpolationMode.BILINEAR), # 28x28 -> 32x32
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.FashionMNIST(root="data",
                                train=True,
                                download=True,
                                transform=transform)
   
    dataloader = DataLoader(dataset=dataset,
                        batch_size=config.BATCH_SIZE,
                        num_workers=config.NUM_WORKERS,
                        shuffle=True
    ) 

    ## Model Initialization 
    netG = Generator(config.NGPU).to(config.DEVICE)
    netD = Discriminator(config.NGPU).to(config.DEVICE)
    netG.apply(weights_init)
    netD.apply(weights_init)

    ## Loss and Optimizers
    criterion = nn.BCELoss()
    optimizerD = torch.optim.Adam(netD.parameters(), lr=config.LEARNING_RATE_D, betas=(config.BETA1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=config.LEARNING_RATE_G, betas=(config.BETA1, 0.999))
    fixed_noise = torch.randn(64, config.NZ, 1, 1, device=config.DEVICE)

    
    ## Training Loop 
    train_time_start = timer()

    print("Starting Training Loop...")
    img_list = []
    G_losses = []
    D_losses = []

    for epoch in range(config.NUM_EPOCHS):
        for i, data in enumerate(dataloader, 0):
            ### Update D network 
            ## Train with all-real batch
            netD.zero_grad()
            real_cpu = data[0].to(config.DEVICE)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), config.REAL_LABEL, dtype=torch.float, device=config.DEVICE)
            output = netD(real_cpu).view(-1)
            lossD_real = criterion(output, label)
            lossD_real.backward()
            D_x = output.mean().item()
            
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, config.NZ, 1, 1, device=config.DEVICE)
            fake = netG(noise)
            label.fill_(config.FAKE_LABEL)
            output = netD(fake.detach()).view(-1)
            lossD_fake = criterion(output, label)
            lossD_fake.backward()
            D_G_z1 = output.mean().item()
            lossD = lossD_real + lossD_fake
            optimizerD.step()
            
            ### Update G network every 2 steps
            if i % 2 == 0:
                netG.zero_grad()
                label.fill_(config.REAL_LABEL)
                output = netD(fake).view(-1)
                lossG = criterion(output, label)
                lossG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()
            else:
                pass    
        
            # Print training results
            if i % 50 == 0:
                print(f'[{epoch+1}/{config.NUM_EPOCHS}][{i}/{len(dataloader)}] Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
        
            # Save losses for plotting later
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())

        # Save generated image grid at the end of each epoch
        with torch.no_grad():
            fake_samples = netG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake_samples, padding=2, normalize=True))
        vutils.save_image(img_list[-1], f"results/fake_samples_epoch_{epoch+1}.png")

    train_time_end = timer()
    total_train_time = train_time_end - train_time_start
    print(f"Total training time: {total_train_time:.3f} seconds")

    ## Visualise the final images and loss curve
    print("Generating final plots...")
    plot_loss_curves(G_losses, D_losses)
    plot_real_vs_fake(config.DEVICE, dataloader, img_list)
    print("Plots saved to the 'results' folder.")


if __name__ == '__main__':
    train()    