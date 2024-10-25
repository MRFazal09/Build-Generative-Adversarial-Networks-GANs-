import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(999)

# Hyperparameters
batch_size = 64
image_size = 64
nz = 100  # Size of the latent vector (input noise to the generator)
ngf = 64  # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator
nc = 1    # Number of channels in the training images (e.g., 1 for grayscale images)
num_epochs = 5
lr = 0.0002  # Learning rate for optimizers
beta1 = 0.5  # Beta1 hyperparam for Adam optimizers

# Data preparation (MNIST dataset)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.MNIST(root='./data', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Create models
netG = Generator()
netG.apply(weights_init)
netD = Discriminator()
netD.apply(weights_init)

# Loss and optimizers
criterion = nn.BCELoss()
fixed_noise = torch.randn(batch_size, nz, 1, 1)

real_label = 1.0
fake_label = 0.0

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training loop
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        ############################
        netD.zero_grad()
        # Train with real batch
        real_data = data[0]
        label = torch.full((real_data.size(0),), real_label, dtype=torch.float)  # Ensure label is float
        output = netD(real_data).view(-1)
        lossD_real = criterion(output, label)
        lossD_real.backward()

        # Train with fake batch
        noise = torch.randn(real_data.size(0), nz, 1, 1)
        fake_data = netG(noise)
        label.fill_(fake_label)
        output = netD(fake_data.detach()).view(-1)
        lossD_fake = criterion(output, label)
        lossD_fake.backward()
        optimizerD.step()

        ############################
        # (2) Update Generator: maximize log(D(G(z)))
        ############################
        netG.zero_grad()
        label.fill_(real_label)  # Generator wants discriminator to believe fake is real
        output = netD(fake_data).view(-1)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

        # Print losses occasionally
        if i % 100 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Step {i}, Loss D: {lossD_real + lossD_fake}, Loss G: {lossG}')

    # Generate samples for visualization
    with torch.no_grad():
        fake = netG(fixed_noise).detach()
    plt.imshow(vutils.make_grid(fake, padding=2, normalize=True).permute(1, 2, 0))
    plt.show()
