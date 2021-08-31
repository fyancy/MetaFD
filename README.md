# Meta-Learning-in-Fault-Diagnosis
The source codes for Meta-learning for few-shot cross-domain fault diagnosis.

# Instructions
1. To run all models, the requirements of your python environmrnt are as: 1) pytorch 1.8.1+cu102; 2) tensorflow-gpu 2.4.0. Note that only `MANN` is implemented by tensorflow, all other methods are achieved by pytorch. Thus, with pytorch only, you can observe the performance of most methods on CWRU dataset.
2. Some packages you have to install: 1) tensorflow_addons (for AdamW in tensorflow. Not really necessary); 2) learn2learn. The latter is an advanced API to achieve meta-learning methods, which is definitely compatible with pytorch. If you have problems when installing learn2learn, such as 'Microsoft Visual C++ 14.0 is required.', please refers to https://zhuanlan.zhihu.com/p/165008313; 3) Visdom (for visualization).
3. The codes of these methods follow the idea of the original paper as far as possible, of course, for application in fault diagnosis, there are some modifications.

# Methods
```python
import torch.nn as nn

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
    )


class encoder_net(nn.Module):
    def __init__(self, in_chn, hidden_chn, cb_num=4):
        super().__init__()
        conv1 = conv_block(in_chn, hidden_chn)
        conv1_more = [conv_block(hidden_chn, hidden_chn) for _ in range(cb_num - 1)]
        self.feature_net = nn.Sequential(conv1, *conv1_more)  # (None, 64, 1024/2^4)

    def forward(self, x):
        return self.feature_net(x)
```
