import torch
from torchvision.datasets import MNIST 
from torchvision import transforms
from scipy import io

def load_MNIST() :
    train_data = MNIST(root = './data/02/',
                                train=True,
                                download=True,
                                transform=transforms.ToTensor())
    test_data = MNIST(root = './data/02/',
                                train=False,
                                download=True,
                                transform=transforms.ToTensor())
    print('number of training data : ', len(train_data))
    print('number of test data : ', len(test_data))
    
    return train_data, test_data

def load_frey_face() :
    file = io.loadmat("frey_rawface.mat")
    file = file["ff"].T.reshape(-1,28,20)
    file = torch.from_numpy(file).float()/255
    
    return file 

