import torch
import torch.nn as nn
import torch.nn.functional as F

def reparameterization(mu, logvar):
    std = torch.exp(logvar/2)
    eps = torch.randn_like(std, dtype=torch.float32)
    return mu + eps * std

#%%
class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()

        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.Tanh()
        )
        nn.init.normal_(self.fc1[0].weight, mean=0, std= 0.1)

        # output layer
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)
        nn.init.normal_(self.mu.weight, mean=0, std= 0.1)
        nn.init.normal_(self.logvar.weight, mean=0, std= 0.1)

    def forward(self, x):
        x = self.fc1(x)

        mu = self.mu(x)
        logvar = self.logvar(x)

        z = reparameterization(mu, logvar)
        return z, mu, logvar
    
#%%
class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()

        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.Tanh()
        )
        nn.init.normal_(self.fc1[0].weight, mean=0, std= 0.1)

        # output layer
        self.fc2 = nn.Linear(h_dim, x_dim)
        nn.init.normal_(self.fc2.weight, mean=0, std= 0.1)

    def forward(self, z):
        z = self.fc1(z)
        x_reconst = torch.sigmoid(self.fc2(z))
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
    
def loss_func(x, x_reconst, mu, logvar):
    kl_div = 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1)
    reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
    loss = kl_div + reconst_loss
    
    return loss
    
class Decoder_norm(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()

        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.Tanh(),
        )
        nn.init.normal_(self.fc1[0].weight, mean=0, std= 0.1)

        # output layer
        self.mu = nn.Linear(h_dim, x_dim)
        self.logvar = nn.Linear(h_dim, x_dim)
        nn.init.normal_(self.mu.weight, mean=0, std= 0.1)
        nn.init.normal_(self.logvar.weight, mean=0, std= 0.1)

    def forward(self, z):
        z = self.fc1(z)
        
        mu_de = torch.sigmoid(self.mu(z))
        logvar_de = self.logvar(z)
        
        x_reconst = reparameterization(mu_de, logvar_de)
        return x_reconst, mu_de, logvar_de
    
class VAE_norm(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()
        self.encoder = Encoder(x_dim, h_dim, z_dim)
        self.decoder = Decoder_norm(x_dim, h_dim, z_dim)
        
    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_reconst, mu_de, logvar_de = self.decoder(z)
        return x_reconst, mu_de, logvar_de, mu, logvar
    
def loss_norm(x, mu_de, logvar_de , mu, logvar):
    kl_div = 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1)
    reconst_loss = 0.5*torch.sum(((x-mu_de)**2)/torch.exp(logvar_de)+logvar_de)
    loss = kl_div + reconst_loss
    
    return loss
