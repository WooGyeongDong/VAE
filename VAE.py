#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy
#%%
config = {'input_dim' : 28*28,
          'hidden_dim' : 500,
          'latent_dim' : 2,
          'batch_size' : 100,
          'epochs' : 30,
          'lr' : 0.01,
          'best_loss' : 10**9,
          'patience_limit' : 3}
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

train_dataloader = DataLoader(train_data, batch_size = 100, shuffle=True )
test_dataloader = DataLoader(test_data, batch_size = 100)
#%%
def reparameterization(mu, logvar):
    std = torch.exp(logvar/2)
    eps = torch.randn_like(std, dtype=torch.float32)
    return mu + eps * std

#%%
class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(Encoder, self).__init__()

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
        super(Decoder, self).__init__()

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
        x_reconst = F.sigmoid(self.fc2(z))
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
model = VAE(x_dim=config['input_dim'], h_dim = config['hidden_dim'], z_dim = config['latent_dim']).to(device)
optimizer = torch.optim.Adagrad(model.parameters(), lr = config['lr'])

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
plt.plot(val)


image, label = test_data[2]
plt.imshow(image.squeeze().numpy(), cmap='gray')
plt.title('label : %s' % label)
plt.show()

x = image.view(-1, img_size)
x = x.to(device)
x_reconst, _, _= model(x)

x = model.decoder(torch.tensor((0.5,0.1)))
plt.imshow(x_reconst.reshape(28,28).detach().numpy(), cmap='gray')
plt.title('label : %s' % label)
plt.show()



# torch.sum(torch.log(1/(((2*torch.pi)**(1/2))*torch.exp(logvar_de/2))*torch.exp(((x-mu_de)**2)/(2*torch.exp(logvar_de)))))

#%%
# class Decoder_norm(nn.Module):
#     def __init__(self, x_dim, h_dim, z_dim):
#         super(Decoder, self).__init__()

#         # 1st hidden layer
#         self.fc1 = nn.Sequential(
#             nn.Linear(z_dim, h_dim),
#             nn.Tanh(),
#         )

#         # output layer
#         self.mu = nn.Linear(h_dim, x_dim)
#         self.logvar = nn.Linear(h_dim, x_dim)


#     def forward(self, z):
#         z = self.fc1(z)
        
#         mu_de = F.relu(self.mu(z))
#         logvar_de = F.relu(self.logvar(z))
        
#         x_reconst = reparameterization(mu_de, logvar_de)
#         return x_reconst, mu_de, logvar_de
