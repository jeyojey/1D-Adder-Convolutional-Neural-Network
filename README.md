# 1D-Adder-Convolutional-Neural-Network

This project implements the AdderNet (Adder Convolutional Neural Network) mentioned in [AdderNet: Do We Really Need Multiplications in Deep Learning?](https://arxiv.org/abs/1912.13200) for 1D sequence processing. Original pytorch implementation of the AdderNet can be found in the [Code Repository](https://github.com/huawei-noah/AdderNet).

The code is adapted to perform 1D additions-based convolutions on a 1D data, i.e. time-series. The 1D sequence processing is widely popular domain with applications from time-series classification to signal filtering. The 1D-AdderNet can be extremely usefull when CNN-level performance is required with reduced computational complexity.

# Data Preprocessing
First step is reshaping the dataset  ...
Here we assume the data equalization task with input sequence RX and the corresponding labels TX with the same shape.

```python
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

syms_in = 17
syms_out = 1
test_percent = 50
RX_train1, RX_test1, y_train1, y_test1 = create_dataset_symbols_multi(syms_in, syms_out, RX, TX, test_percent)

RX_train1 = torch.tensor(RX_train1.reshape((len(RX_train1),1,syms_in,1)), dtype=torch.float32).to(device)
RX_test1 = torch.tensor(RX_test1.reshape((len(RX_test1),1,syms_in,1)), dtype=torch.float32).to(device)
y_train1 = torch.tensor(y_train1, dtype=torch.float32).to(device)
```

# Build 1D-AdderCNN


# References 
[1] [AdderNet: Do We Really Need Multiplications in Deep Learning?](https://arxiv.org/abs/1912.13200)

[2] [Code Repository](https://github.com/huawei-noah/AdderNet)
