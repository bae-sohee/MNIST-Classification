
import dataset
from model import LeNet5, CustomMLP
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in trn_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    trn_loss = total_loss / len(trn_loader)
    acc = 100. * correct / total

    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data, target in tst_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    tst_loss = total_loss / len(tst_loader)
    acc = 100. * correct / total

    return tst_loss, acc
  
def main(model_type='LeNet5', epochs=10, lr=0.01, batch_size=64):
    """ Main function

        This process involves creating dataset objects, setting up data loaders, and initializing models, It includes setting up an optimization method, defining a loss function, and so on.
        
        1) Dataset objects for training and test datasets 
        2) DataLoaders for training and testing 
        3) Set up the calculation unit (CPU or GPU) and initialize the model
        4) Using SGD as an optimizer : initial learning rate 0.01 and momentum 0.9
        5) CrossEntropy loss as the loss function
        6) Initializes the list to store the loss and accuracy
        7) Train and test the model for each epoch

    """
    # 1)
    train_dir = '/home/user/Desktop/bsh/DL/data/train'
    test_dir = '/home/user/Desktop/bsh/DL/data/test'
    trn_dataset = dataset.MNIST(train_dir)
    tst_dataset = dataset.MNIST(test_dir)
    
    # 2)
    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
    tst_loader = DataLoader(tst_dataset, batch_size=batch_size, shuffle=False)
    
    # 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == 'LeNet5':
        model = LeNet5().to(device)
    elif model_type == 'CustomMLP':
        model = CustomMLP().to(device)
    else:
        raise ValueError("Unsupported model type")

    # 4)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # 5)
    criterion = nn.CrossEntropyLoss()

    # 6)
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    # 7)
    for epoch in range(1, epochs+1):
        trn_loss, trn_accuracy = train(model, trn_loader, device, criterion, optimizer)
        tst_loss, tst_accuracy = test(model, tst_loader, device, criterion)
        print(f'Epoch: {epoch}, Train Loss: {trn_loss:.4f}, Train Accuracy: {trn_accuracy:.2f}, Test Loss: {tst_loss:.4f}, Test Accuracy: {tst_accuracy:.2f}')

        train_losses.append(trn_loss)
        train_accuracies.append(trn_accuracy)
        test_losses.append(tst_loss)
        test_accuracies.append(tst_accuracy)

    return train_losses, train_accuracies, test_losses, test_accuracies

def save_plots(train_losses, train_accuracies, test_losses, test_accuracies, model):
    """Save_plots function

        Create and store loss and accuracy graphs for training and testing datasets

    Args: 
        train_losses: training loss values by epoch
        train_accuracies: training accuracy by epoch
        test_losses: test loss values by epoch
        test_accuracies: test accuracy by epoch
        model: Instance of the model to be used for the title of the graph
    
    Outputs: 
        Save graphs of loss and accuracy for train and test values (2)
    
    """
    model_name = model.__class__.__name__  
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title(f'Loss over epochs_{model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_plot_{model_name}.png')  
    plt.close()  

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title(f'Accuracy over epochs_{model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'accuracy_plot_{model_name}.png')  
    plt.close()  

if __name__ == '__main__':

    # args
    model_type = 'CustomMLP'  # 'CustomMLP' or 'LeNet5'
    epochs = 20
    lr = 0.001
    batch_size = 64

    train_losses, train_accuracies, test_losses, test_accuracies = main(model_type, epochs, lr, batch_size)
    model = CustomMLP() if model_type == 'CustomMLP' else LeNet5()
    save_plots(train_losses, train_accuracies, test_losses, test_accuracies, model)