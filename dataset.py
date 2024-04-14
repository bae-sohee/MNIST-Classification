import os
import tarfile

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import utils
from PIL import Image

import matplotlib.pyplot as plt

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """
        
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.filenames = os.listdir(data_dir)
        # Set up the transformation
        self.transform = Compose([
            ToTensor(),                      # [0,1] scaling
            Normalize((0.1307,), (0.3081,))  # Normalizeing
        ])

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filepath = os.path.join(self.data_dir, self.filenames[idx])
        img = Image.open(filepath).convert('L') 
        img = self.transform(img)   
        label = int(self.filenames[idx].split('_')[1].split('.')[0]) # labels in filename
        return img, label
    
def extract_tar_file(tar_path, extract_path):
    """
    Function to unzip a tar file to a given path
    :param tar_path: Path to the tar file to unpack.
    :param extract_path: Path to save the uncompressed files.
    """
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    with tarfile.open(tar_path, "r:") as tar:
        tar.extractall(path=extract_path)

if __name__ == '__main__':

    # dataset tar file path
    train_tar_path = '/home/user/Desktop/bsh/DL/data/train.tar'
    test_tar_path = '/home/user/Desktop/bsh/DL/data/test.tar'

    # save path
    extract_path = '/home/user/Desktop/bsh/DL/data'

    # unzip tar
    extract_tar_file(train_tar_path, extract_path)
    extract_tar_file(test_tar_path, extract_path)

    # MNIST dataset instance
    train_dataset = MNIST(os.path.join(extract_path, 'train'))
    test_dataset = MNIST(os.path.join(extract_path, 'test'))

    # dataset length 
    print(f"train dataset length: {len(train_dataset)}")
    print(f"test dataset length: {len(test_dataset)}")
