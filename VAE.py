#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim 
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy
import wandb
import importlib
from data import load_MNIST
from model_class import *
#%%
config = {'input_dim' : 28*28,
          'hidden_dim' : 500,
          'latent_dim' : 10,
          'batch_size' : 100,
          'epochs' : 3,
          'lr' : 0.01,
          'best_loss' : 10**9,
          'patience_limit' : 3}
#%%
wandb.init(project="VAE", config=config)
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
print('Current cuda device is', device)

#%%
train_data, test_data = load_MNIST()

train_dataloader = DataLoader(train_data, batch_size = config['batch_size'], shuffle=True )
test_dataloader = DataLoader(test_data, batch_size = config['batch_size'])

#%%
model = VAE(x_dim=config['input_dim'], h_dim = config['hidden_dim'], z_dim = config['latent_dim']).to(device)
optimizer = torch.optim.Adagrad(model.parameters(), lr = config['lr'])

model_n = VAE_norm(x_dim=config['input_dim'], h_dim = config['hidden_dim'], z_dim = config['latent_dim']).to(device)
optimizer_n = torch.optim.Adagrad(model_n.parameters(), lr = config['lr'])

#%%
def loss_func(x, x_reconst, mu, logvar):
    kl_div = 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1)
    reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
    loss = kl_div + reconst_loss
    
    return loss

#%% 
img_size = config['input_dim']  
best_loss = config['best_loss']
patience_limit = config['patience_limit']
patience_check = 0 # 현재 몇 epoch 연속으로 loss 개선이 안되는지를 기록
val = []
for epoch in tqdm(range(config['epochs'])):
    model.train()
    train_loss = 0
    for x, _ in train_dataloader:
        # forward
        x = x.view(-1, img_size)
        x = x.to(device)
        x_reconst, mu, logvar = model(x)

        # backprop and optimize
        loss = loss_func(x, x_reconst, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print('Epoch: {} Train_Loss: {} :'.format(epoch, train_loss/len(train_dataloader.dataset)))    
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_val, _ in test_dataloader:
            x_val = x_val.view(-1, img_size)
            x_val = x_val.to(device)
            x_val_reconst, mu, logvar = model(x_val)

            loss = loss_func(x_val, x_val_reconst, mu, logvar).item()
            val_loss += loss/len(test_dataloader.dataset)
        val.append(val_loss)
        wandb.log({'train_loss':train_loss/len(train_dataloader.dataset), 'valid_loss': val_loss})
        print(epoch, val_loss)
        if abs(val_loss - best_loss) < 1e-3: # loss가 개선되지 않은 경우
            patience_check += 1

            if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
                print("Learning End. Best_Loss:{:6f}".format(best_loss))
                break

        else: # loss가 개선된 경우
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            patience_check = 0
    

# %%
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
        
        mu_de = self.mu(z)
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

model_n = VAE_norm(x_dim=config['input_dim'], h_dim = config['hidden_dim'], z_dim = config['latent_dim']).to(device)
optimizer_n = torch.optim.Adagrad(model_n.parameters(), lr = 0.01)

#%%

img_size = config['input_dim']  
best_loss = config['best_loss']
patience_limit = config['patience_limit']
patience_check = 0 # 현재 몇 epoch 연속으로 loss 개선이 안되는지를 기록
val = []
for epoch in tqdm(range(config['epochs'])):
    model_n.train()
    train_loss = 0
    for x, _ in train_dataloader:
        # forward
        x = x.view(-1, img_size)
        x = x.to(device)
        x_reconst, mu_de, logvar_de, mu, logvar = model_n(x)

        # backprop and optimize
        loss = loss_norm(x, mu_de, logvar_de, mu, logvar)
        optimizer_n.zero_grad()
        loss.backward()
        optimizer_n.step()
        train_loss += loss.item()
    print('Epoch: {} Train_Loss: {} :'.format(epoch, train_loss/len(train_dataloader.dataset)))    
    
    model_n.eval()
    val_loss = 0
    with torch.no_grad():
        for x_val, _ in test_dataloader:
            x_val = x_val.view(-1, img_size)
            x_val = x_val.to(device)
            x_val_reconst, mu_de, logvar_de, mu, logvar = model_n(x_val)

            loss = loss_norm(x, mu_de, logvar_de, mu, logvar).item()
            val_loss += loss/len(test_dataloader.dataset)
        val.append(val_loss)

        print(epoch, val_loss)
        if abs(val_loss - best_loss) < 1e-3: # loss가 개선되지 않은 경우
            patience_check += 1

            if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
                print("Learning End. Best_Loss:{:6f}".format(best_loss))
                break

        else: # loss가 개선된 경우
            best_loss = val_loss
            best_model = copy.deepcopy(model_n)
            patience_check = 0



#%%
plt.plot(val)


image, label = test_data[3]
plt.imshow(image.squeeze().numpy())
plt.title('label : %s' % label)
plt.show()

x = image.view(-1, img_size)
x = x.to(device)
x_reconst,_,_, _, _= model_n(x)

# x = model.decoder(torch.tensor((0.5,0.1)))
plt.imshow(x_reconst.reshape(28,28).detach().numpy())
plt.title('label : %s' % label)
plt.show()

#%%

torch.eye(2)
