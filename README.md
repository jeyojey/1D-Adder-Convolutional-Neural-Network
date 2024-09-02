# 1D-Adder-Convolutional-Neural-Network

This project implements the AdderNet (Adder Convolutional Neural Network) mentioned in [AdderNet: Do We Really Need Multiplications in Deep Learning?](https://arxiv.org/abs/1912.13200) for 1D sequence processing. Original pytorch implementation of the AdderNet can be found in the [Code Repository](https://github.com/huawei-noah/AdderNet).

The code is adapted to perform 1D additions-based convolutions on a 1D data, i.e. time-series. The 1D sequence processing is widely popular domain with applications from time-series classification to signal filtering. The 1D-AdderNet can be extremely usefull when CNN-level performance is required with reduced computational complexity.


# References 
[1] [AdderNet: Do We Really Need Multiplications in Deep Learning?](https://arxiv.org/abs/1912.13200)

[2] [Code Repository](https://github.com/huawei-noah/AdderNet)
