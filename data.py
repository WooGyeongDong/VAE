from torchvision.datasets import MNIST 
from torchvision import transforms

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
