# MNIST Classification
- Building a Neural Network Classifier with MNIST Dataset

### Assignment Objective
- For the MNIST, we implement a data loader and a classification model.
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
- Each of tar files contains 60,000 training images and 10,000 test images respectively
- Each image has its own filename like `{ID}_{Label}.png`
- Run `dataset.py` to extract tar compressed files
  ```bash
  ├── data
  │   ├── train
  │   │    └── 00000_5.png
  │   │    └── 00001_0.png
  │   │        ...
  │   ├── test
  │   │    └── 00000_7.png
  │   │    └── 00001_2.png
  │   │        ...
```
## 3. Implementation
- You need to run `main.py`.

  ```bash
  python main.py
  ```
- Model training configuration can be set in #args
- The default settings are as follows
    # args
    model_type = 'LeNet5'  
    epochs = 10
    lr = 0.01
    batch_size = 64

## 4. Model Structure
- You can check the structure of the model by running `model.py`.

  ```bash
  python model.py
  ```

LeNet-5 Summary:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 6, 24, 24]             156
         MaxPool2d-2            [-1, 6, 12, 12]               0
            Conv2d-3             [-1, 16, 8, 8]           2,416
         MaxPool2d-4             [-1, 16, 4, 4]               0
            Linear-5                  [-1, 120]          30,840
            Linear-6                   [-1, 84]          10,164
            Linear-7                   [-1, 10]             850
================================================================
Total params: 44,426
Trainable params: 44,426
Non-trainable params: 0
----------------------------------------------------------------

CustomMLP Summary:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                   [-1, 56]          43,960
            Linear-2                   [-1, 28]           1,596
            Linear-3                   [-1, 10]             290
================================================================
Total params: 45,846
Trainable params: 45,846
Non-trainable params: 0
----------------------------------------------------------------

- How to calculate the number of model parameters of LeNet5 and CustomMLP

- LeNet-5
1. Conv2d-1(conv1) 
     input channel : 1, output channel : 6, kernal_size : 5*5, bias : 6
     total parms : (5*5*1*6)+6 = 156
2. Conv2d-2(conv2)
     input channel : 6, output channel : 16, kernal_size : 5*5, bias : 16
     total parms : (5*5*6*16)+16 = 2,146
3. Linear-1(fc1)
     input channel : 256, output channel : 120, bias : 120
     total parms : (256*120)+120 = 30,840
4. Linear-2(fc2)
     input channel : 120, output channel : 84, bias : 84
     total parms : (120*84)+84 = 10,164
5. Linear-3(fc3)
     input channel : 84, output channel : 10, bias : 10
     total parms : (84*10)+10 = 850

     Total parameters of LeNet-5 = 156 + 2,146 + 30,840 + 10,164 + 850 = 44,426

- CustomMLP
1. Linear-1(fc1)
     input channel : 784, output channel : 56, bias : 56
     total parms : (784*56)+56 = 43,960
2. Linear-2(fc2)
     input channel : 56, output channel : 28, bias : 28
     total parms : (56*28)+28 = 1,596
3. Linear-3(fc3)
     input channel : 28, output channel : 10, bias : 10
     total parms : (28*10)+10 = 290

     Total parameters of CustomMLP = 43,960 + 1,596 + 290 = 45,846

## 5. Result

## 6. Refecence

LeNet5 모델 : https://deep-learning-study.tistory.com/503
tar 파일 압축 풀기 : https://salguworld.tistory.com/entry/Python-tarfile-%EB%AA%A8%EB%93%88%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%9C-tar-%EC%95%95%EC%B6%95-%ED%95%B4%EC%A0%9C-%EB%B0%8F-%ED%8C%8C%EC%9D%BC-%EB%AA%A9%EB%A1%9D-%ED%99%95%EC%9D%B8%ED%95%98%EA%B8%B0-1
