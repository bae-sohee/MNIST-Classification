import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998), the CNN architectures

        - a total of seven layers : Convolutional layers(3), Max pooling(3), Fully connected layers(2)
        - total params: 44,426
         
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4, stride=1, padding=0)      
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10) # output =  num_class

    def forward(self, img):
        img = F.relu(self.conv1(img))
        img = self.pool(img)
        img = F.relu(self.conv2(img))
        img = self.pool(img)
        img = F.relu(self.conv3(img))
        img = img.view(-1, 120)
        img = F.relu(self.fc1(img))
        output = self.fc2(img)

        return output
    
class LeNet5_regularization(nn.Module):
    """ LeNet-5 (LeCun et al., 1998), the CNN architectures

        - a total of seven layers : Convolutional layers(3), Max pooling(3), Fully connected layers(2)
        - total params: 44,426
         
    """
    def __init__(self):
        super(LeNet5_regularization, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)      # Batch Normalization
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)     # Batch Normalization
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4, stride=1, padding=0)   
        self.bn3 = nn.BatchNorm2d(120)   
        self.fc1 = nn.Linear(120, 84)
        self.dropout1 = nn.Dropout(p=0.5)  # Dropout
        self.fc2 = nn.Linear(84, 10) # output = num_class
        
    def forward(self, img):
        img = F.relu(self.conv1(img))
        img = self.pool(img)
        img = F.relu(self.conv2(img))
        img = self.pool(img)
        img = F.relu(self.conv3(img))
        img = img.view(-1, 120)
        img = F.relu(self.fc1(img))
        output = self.fc2(img)

        return output

class CustomMLP(nn.Module):
    """ CustomMLP model

        - a simple multilayer perceptron model consisting of fully connected layers.
        - a total of three layers : Fully connected layers(3)
        - total params: 45,846
        
    """
    def __init__(self):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 56)
        self.fc2 = nn.Linear(56, 28)
        self.fc3 = nn.Linear(28, 10) # output =  num_class
        
    def forward(self, img):
        img = img.view(-1, 28*28)
        img = F.relu(self.fc1(img))
        img = F.relu(self.fc2(img))
        output = self.fc3(img)

        return output

if __name__ == '__main__':

    model_lenet5 = LeNet5()
    model_custommlp = CustomMLP()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_lenet5.to(device)
    model_custommlp.to(device)

    print("LeNet-5 Summary:")
    summary(model_lenet5, (1, 28, 28))

    print("CustomMLP Summary:")
    summary(model_custommlp, (1, 28*28)) 

    