#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import copy
#%%
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
print('Current cuda device is', device)

#%%
train_data = datasets.MNIST(root = './data/02/',
                            train=True,
                            download=True,
                            transform=transforms.ToTensor())
test_data = datasets.MNIST(root = './data/02/',
                            train=False,
                            download=True,
                            transform=transforms.ToTensor())
print('number of training data : ', len(train_data))
print('number of test data : ', len(test_data))

# %%

image, label = train_data[0]
plt.imshow(image.squeeze().numpy(), cmap='gray')
plt.title('label : %s' % label)
plt.show()

train_dataloader = DataLoader(train_data, batch_size = 512, shuffle=True )
#%%
def reparameterization(mu, logvar):
    std = torch.exp(logvar/2)
    eps = torch.randn_like(std)
    return mu + eps * std

#%%
class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(Encoder, self).__init__()

        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # 2nd hidden layer
        self.fc2 = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # output layer
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        x = self.fc2(self.fc1(x))

        mu = F.relu(self.mu(x))
        logvar = F.relu(self.logvar(x))

        z = reparameterization(mu, logvar)
        return z, mu, logvar
    
#%%
class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(Decoder, self).__init__()

        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        # 2nd hidden layer
        self.fc2 = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # output layer
        self.fc3 = nn.Linear(h_dim, x_dim)

    def forward(self, z):
        z = self.fc2(self.fc1(z))
        x_reconst = F.sigmoid(self.fc3(z))
        return x_reconst

#%%
class VAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()
        self.encoder = Encoder(x_dim, h_dim, z_dim)
        self.decoder = Decoder(x_dim, h_dim, z_dim)
        
    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_reconst = self.decoder(z)
        return x_reconst, mu, logvar

#%%
img_size = 28**2
hidden_dim = 256
latent_dim = 2

model = VAE(img_size, hidden_dim, latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

#%%   
n_epochs = 50
for epoch in tqdm(range(n_epochs)):
    for i, (x, _) in enumerate(train_dataloader):
        # forward
        x = x.view(-1, img_size)
        x = x.to(device)
        x_reconst, mu, logvar = model(x)

        # compute reconstruction loss and KL divergence
        reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
        kl_div = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)

        # backprop and optimize
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# %%
image, label = test_data[0]
plt.imshow(image.squeeze().numpy(), cmap='gray')
plt.title('label : %s' % label)
plt.show()
image.squeeze().shape
x = image.view(-1, img_size)
x = x.to(device)
x_reconst, mu, logvar = model(x)

plt.imshow(x_reconst.reshape(28,28).detach().numpy(), cmap='gray')
plt.title('label : %s' % label)
plt.show()