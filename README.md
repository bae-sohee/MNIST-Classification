# MNIST Classification
- Building a Neural Network Classifier with MNIST Dataset

### Assignment Objective
- For the MNIST, We implement a data loader and a classification model.
- We obtain the CNN model and the MLP model and compare the performance and loss of the model.

## 1. Environment
- Python version is 3.8.
- Used 'PyTorch' and device type as 'GPU'.
- `requirements.txt` file is required to set up the virtual environment for running the program. This file contains a list of all the libraries needed to run your program and their versions.

    #### In **Anaconda** Environment,

  ```
  $ conda create -n [your virtual environment name] python=3.9
  
  $ conda activate [your virtual environment name]
  
  $ pip install -r requirements.txt
  ```

  - Create your own virtual environment.
  - Activate your Anaconda virtual environment where you want to install the package. If your virtual environment is named 'test', you can type **conda activate test**.
  - Use the command **pip install -r requirements.txt** to install libraries.

## 2. Dataset
- Each of tar files contains 60,000 training images and 10,000 test images respectively.
- Each image has its own filename like `{ID}_{Label}.png`.
- Run `dataset.py` to extract .tar compressed files.

  ```bash
  python dataset.py
  ```

  ```bash
  |-- data
  |   |-- train
  |   |   |- 00000_5.png
  |   |   |- 00001_0.png
  |   |   |- ...
  |   |
  |   |-- test
  |       |- 00000_7.png
  |       |- 00001_2.png
  |       |- ...
  ```

## 3. Implementation
- You need to run `main.py`.
  ```bash
  python main.py
  ```
- Model training configuration can be set in **args**.
- The default settings are as follows.
    #### args
    model_type = 'LeNet5'  
    epochs = 20  
    batch_size = 64  

## 4. Model Structure
- You can check the structure of the model by running `model.py`.

  ```bash
  python model.py
  ```

### LeNet-5 Summary

| Layer (type)   | Output Shape      | Param # |
|----------------|-------------------|---------|
| Conv2d-1       | [-1, 6, 24, 24]   | 156     |
| MaxPool2d-2    | [-1, 6, 12, 12]   | 0       |
| Conv2d-3       | [-1, 16, 8, 8]    | 2,416   |
| MaxPool2d-4    | [-1, 16, 4, 4]    | 0       |
| Conv2d-5       | [-1, 120, 1, 1]   | 30,840  |
| Linear-6       | [-1, 84]          | 10,164  |
| Linear-7       | [-1, 10]          | 850     |
| **Total**      |                   | **44,426** |

### CustomMLP Summary

| Layer (type)   | Output Shape      | Param # |
|----------------|-------------------|---------|
| Linear-1       | [-1, 56]          | 43,960  |
| Linear-2       | [-1, 28]          | 1,596   |
| Linear-3       | [-1, 10]          | 290     |
| **Total**      |                   | **45,846** |

### How to calculate the number of model parameters of LeNet5 and CustomMLP

- **LeNet-5**
1. Conv2d-1(conv1) 
     input channel : 1, output channel : 6, kernal_size : 5 * 5, bias : 6  
     total parms : (5 * 5 * 1 * 6) + 6 = 156
2. Conv2d-2(conv2)
     input channel : 6, output channel : 16, kernal_size : 5 * 5, bias : 16  
     total parms : (5 * 5 * 6 * 16) + 16 = 2,416
3. Conv2d-2(conv3)
     input channel : 16, output channel : 120, kernal_size : 4 * 4, bias : 120  
     total parms : (4 * 4 * 16 * 120) + 120 = 30,840
4. Linear-2(fc2)
     input channel : 120, output channel : 84, bias : 84  
     total parms : (120 * 84) + 84 = 10,164
5. Linear-3(fc3)
     input channel : 84, output channel : 10, bias : 10  
     total parms : (84 * 10) + 10 = 850

     Total parameters of LeNet-5 = 156 + 2,416 + 30,840 + 10,164 + 850 = 44,426

- **CustomMLP**
1. Linear-1(fc1)
     input channel : 784, output channel : 56, bias : 56  
     total parms : (784 * 56) + 56 = 43,960
2. Linear-2(fc2)
     input channel : 56, output channel : 28, bias : 28  
     total parms : (56 * 28) + 28 = 1,596
3. Linear-3(fc3)
     input channel : 28, output channel : 10, bias : 10  
     total parms : (28 * 10) + 10 = 290

     Total parameters of CustomMLP = 43,960 + 1,596 + 290 = 45,846

## 5. Result

- Accuracy for each model
  - CustomMLP
  ![acc_plot_CustomMLP](https://github.com/bae-sohee/MNIST_Classification/assets/123538321/70600fa9-4b39-44fb-a30f-6b00f621f1cf)
  - LeNet-5
  ![acc_plot_LeNet-5](https://github.com/bae-sohee/MNIST_Classification/assets/123538321/329c37ca-5a73-4612-be2d-38db9d2ec3cd)
  - LeNet-5 (regularization)
  ![acc_plot_LeNet-5_regularization](https://github.com/bae-sohee/MNIST_Classification/assets/123538321/7bd85219-d64d-40de-a451-774aa7407107)

- Loss for each model
  - CustomMLP
  ![loss_plot_CustomMLP](https://github.com/bae-sohee/MNIST_Classification/assets/123538321/e0be35c8-21dd-4be8-929a-e36f9f793ed8)
  - LeNet-5
  ![loss_plot_LeNet5](https://github.com/bae-sohee/MNIST_Classification/assets/123538321/a2b00775-09ff-40bd-85d4-eaff0f860855)
  - LeNet-5 (regularization)
  ![loss_plot_LeNet5_regularization](https://github.com/bae-sohee/MNIST_Classification/assets/123538321/2708a020-8b51-4224-8f77-bd919aec7da7)

- As a result of comparing the performance of the Custom MLP and LeNet-5 models through 20 epoch, the result were 97.66 for Custom MLP and 98.80 for LeNet-5. Although the similar parameters of the two models (Custom MLP: 45,846, LeNet-5: 44,426), the CNN-based model performs better than the MLP model.
- From checking the learning curves of both models, the loss decreases exponentially as the learning progresses, and it seems to converge from 0.10-0.15 for Custom NLP and 0.04-0.06 for LeNet-5.
- To improve the LeNet-5 model performance, two regularization techniques were applied: Batch normalization and Dropout. The performance was 98.80 for LeNet-5, and 99.13 for LeNet-5 with regularization, showing very slight performance improvement. Due to the low complexity of the data or model used in the experiment, it is assumed that the performance difference between the two models did not appear significantly.  
- For the known accuracy of the existing LeNet-5, the reference result was referred to as reference result. The reference result (about 97.5, 97.64)

## 6. Refecence

LeNet5 model  
https://deep-learning-study.tistory.com/503  
https://deep-learning-study.tistory.com/368?category=963091  

Train  
https://velog.io/@skarb4788/%EB%94%A5-%EB%9F%AC%EB%8B%9D-MNIST-%EB%8D%B0%EC%9D%B4%ED%84%B0PyTorch

Unzip the .tar file  
https://salguworld.tistory.com/entry/Python-tarfile-%EB%AA%A8%EB%93%88%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%9C-tar-%EC%95%95%EC%B6%95-%ED%95%B4%EC%A0%9C-%EB%B0%8F-%ED%8C%8C%EC%9D%BC-%EB%AA%A9%EB%A1%9D-%ED%99%95%EC%9D%B8%ED%95%98%EA%B8%B0-1  
