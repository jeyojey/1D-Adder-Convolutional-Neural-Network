# 1D-Adder-Convolutional-Neural-Network

This project implements the AdderNet (Adder Convolutional Neural Network) mentioned in [AdderNet: Do We Really Need Multiplications in Deep Learning?](https://arxiv.org/abs/1912.13200) for 1D sequence processing. Original pytorch implementation of the AdderNet can be found in the [Code Repository](https://github.com/huawei-noah/AdderNet).

The code is adapted to perform 1D additions-based convolutions on a 1D data, i.e. time-series. The 1D sequence processing is widely popular domain with applications from time-series classification to signal filtering. The 1D-AdderNet can be extremely usefull when CNN-level performance is required with reduced computational complexity.

# Data Preprocessing
First step is reshaping the dataset  ...
Here we assume the data equalization task with input sequence RX and the corresponding labels TX with the same shape.

```python
# RX is a numpy array of recieved symbols (with noise)
# TX is a numpy array of transmitted symbols (labels)

import torch
import create_dataset_symbols_multi
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create dataset with memory shifting each step by 1 symbol
steps = 8                   # number of previous or forward steps to define the memory window
syms_in = steps*2 + 1       # the total size of memory window
syms_out = 1                # number of outputs
test_percent = 50
RX_train, RX_test, y_train, y_test = create_dataset_symbols_multi(syms_in, syms_out, RX, TX, test_percent)

RX_train = torch.tensor(RX_train.reshape((len(RX_train), 1, syms_in, 1)), dtype=torch.float32).to(device)
RX_test = torch.tensor(RX_test.reshape((len(RX_test), 1, syms_in, 1)), dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
#y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
```

# Build 1D-AdderCNN

Here is the example of building the 1D-AdderCNN with 2 hidden layers based on additions. Moreover, the final output layer (linear) is built on the concept of additions-based single convolution, to completely avoid between-layers multiplications.

```python
import AdderNet1D
import torch
import torch.nn as nn

class AdderNet1D(nn.Module):
    def __init__(self):
      super(AdderNet1D, self).__init__()
      filters1 = 15
      f_size1 = 5
      filters2 = 15
      f_size2 = 5

      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      ################################################################################################################################
      self.adder1 = adder.adder1d(input_channel=1, output_channel=filters1, kernel_size=f_size1, stride=1, bias=True).to(self.device)
      self.batch1 = nn.BatchNorm2d(filters1).to(self.device)
      self.acti1 = nn.Sigmoid()
      ################################################################################################################################
      self.adder2 = adder.adder1d(input_channel=filters1, output_channel=filters2, kernel_size=f_size2, stride=1, bias=True).to(self.device)
      self.batch2 = nn.BatchNorm2d(filters2).to(self.device)
      self.acti2 = nn.Sigmoid()
      ################################################################################################################################
      self.linear = adder.adder1d(input_channel=filters2, output_channel=1, kernel_size=int((syms_in - f_size1 + 1) - f_size2 + 1), stride=1, bias=True).to(self.device)
      self.batch3 = nn.BatchNorm2d(1).to(self.device)
      self.acti_out = nn.Sigmoid()
      self.flat = nn.Flatten()

    def forward(self, x):
      x = self.adder1(x)
      x = self.batch1(x)
      x = self.acti1(x)

      x = self.adder2(x)
      x = self.batch2(x)
      x = self.acti2(x)

      x = self.linear(x)
      x = self.batch3(x)
      x = self.acti_out(x)
      x = self.flat(x)

      return x


```

# References 
[1] [AdderNet: Do We Really Need Multiplications in Deep Learning?](https://arxiv.org/abs/1912.13200)

[2] [Code Repository](https://github.com/huawei-noah/AdderNet)
