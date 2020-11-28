# lambda.pytorch

PyTorch implementation of [LambdaNetworks: Modeling long-range Interactions without Attention](https://openreview.net/forum?id=xTJEN-ggl1b).

Lambda Networks apply associative law of matrix multiplication to reverse the computing order of self-attention, achieving the linear computation complexity regarding content interactions.

Similar techniques have been used previously in [A<sup>2</sup>-Net](https://arxiv.org/abs/1810.11579) and [CGNL](https://arxiv.org/abs/1810.13125). Check out a collection of self-attention modules in another repository [dot-product-attention](https://github.com/d-li14/dot-product-attention).

## Training Configuration
✓ SGD optimizer, initial learning rate 0.1, momentum 0.9, weight decay 0.0001

✓ epoch 130, batch size 256, 8x Tesla V100 GPUs, LR decay strategy cosine

✓ label smoothing 0.1

## Pre-trained checkpoints
| Architecture             | Parameters | FLOPs | Top-1 / Top-5 Acc. (%) | Download |
| :----------------------: | :--------: | :---: | :------------------------: | :------: |
| Lambda-ResNet-50 | 14.995M | 6.576G | 78.208 / 93.820 | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EUZkICtpXitIq6PGa6h6m_YBnFXCiCYTSuqoIUqiR33C5A?e=mhgEbC) &#124; [log](https://hkustconnect-my.sharepoint.com/:t:/g/personal/dlibh_connect_ust_hk/EQuZ1itCS2dFpN2MBVepL5YBQe9N-ZUv6y4vNdO5uiVFig?e=dX7Id1) |
