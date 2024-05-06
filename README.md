<a href="https://blog.csdn.net/weixin_43543177/article/details/119974019?spm=1001.2014.3001.5501" target="_blank">
 <img align="right" src="/Results_png/head_clock.png" width="15%"/>
</a>

# Meta-Learning-in-Fault-Diagnosis
![](https://img.shields.io/badge/language-python-orange.svg)
[![](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/fyancy/MetaFD/blob/main/LICENSE)
[![](https://img.shields.io/badge/CSDN-燕策西-blue.svg)](https://blog.csdn.net/weixin_43543177?spm=1001.2101.3001.5343)
[![](https://img.shields.io/badge/Homepage-YongFeng-purple.svg)](https://fyancy.github.io/)
<!-- 如何设计GitHub badge: https://lpd-ios.github.io/2017/05/03/GitHub-Badge-Introduction/ -->

The source codes for Meta-learning in few-shot cross-domain fault diagnosis. Based on this work, our article [Meta-learning as a promising approach for few-shot cross-domain fault diagnosis: Algorithms, applications, and prospects](https://www.sciencedirect.com/science/article/abs/pii/S0950705121009084?via%3Dihub) has been published.

# 1. Instructions
* To run all models, the requirements of your python environmrnt are as: 1) pytorch 1.8.1+cu102; 2) tensorflow-gpu 2.4.0. Note that only `MANN` is implemented by tensorflow, all other methods are achieved by pytorch. Thus, with pytorch only, you can observe the performance of most methods on CWRU dataset.
* Some packages you have to install: 1) tensorflow_addons (for optimizer AdamW in tensorflow. Not really necessary); 2) [learn2learn](https://github.com/learnables/learn2learn). The latter is an advanced API to achieve meta-learning methods, which is definitely compatible with pytorch. If you have problems when installing learn2learn, such as 'Microsoft Visual C++ 14.0 is required.', please refer to [this blog](https://zhuanlan.zhihu.com/p/165008313). Also, You can refer to [this blog](https://blog.csdn.net/weixin_43543177/article/details/119974019) for quick start ; 3) Visdom (for visualization).
* Note that the learn2learn version we used is 0.1.5, if you have issues when using learn2learn, you can use this version or a higher version (>=0.1.5). Or you can modify the codes as https://github.com/fyancy/MetaFD/issues/1
* Change the data path in `cwru_path.py` to put your own `root_dir` of CWdata_12k.
* The codes of these methods follow the idea of the original paper as far as possible, of course, for application in fault diagnosis, there are some modifications.

# 2. Methods
```
1. CNN
2. CNN with fine-tuning (CNN-FT) [1]
3. CNN with Maximum Mean Discrepancy (CNN-MMD) [2]
4. Model Agnostic Meta-Learning (MAML) [3]
5. Reptile [4]
6. Memory Augmented Neural Network (MANN) [5]
7. Prototypical Networks (ProtoNet) [6]
8. Relation Networks (RelationNet) [7]
```
**NOTE**: You can get [**weights** of all well-Trained models](https://drive.google.com/drive/folders/1leHVoYXpMVXM_e148KmBWVaX0WkTZfO6?usp=sharing) now.
- Google Drive
地址：https://drive.google.com/drive/folders/1leHVoYXpMVXM_e148KmBWVaX0WkTZfO6?usp=sharing
- BaiduNetDisk
链接：https://pan.baidu.com/s/1tyj3B7CuM9Tu1-WcizAAbQ 
提取码：oe9p


## 2.1 Feature extractor
The backbone of these methods, i.e. feature extractor, consists of four convolution blocks, as follows
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
## 2.2 Tasks on CWRU bearing dataset
```
T1: 10 ways, load 3 ==> 10 ways, load 0  
T2: 6 ways, load 0 ==> 4 ways, load 0  
Details can be found in `cwru_path.py` 
```
|Tasks|Source categories|Target categories|Source load|Target load|
|:---:|---|---|:---:|:---:|
|T1|{NC, IF1, IF2, IF3, ..., RoF3 }|	{NC, IF1, IF2, IF3, ..., RoF3}|	3|	0|
|T2|{IF1, IF2, IF3, OF1, OF2, OF3}|	{NC, RoF1, RoF2, RoF3 }|	0|	0|

## 2.3 Results (Click on the image to see details)
|Fig. 1. Results on T1.   | Fig. 2. Results on T2.  | Fig. 3. Test time and model memory.  |
|:----:|:----:|:----:|
|<img src="/Results_png/090113071399_0T1_5_1shot_1.Jpeg" width="300" /><br/> | <img src="/Results_png/090113074881_0T2_5_1shot_1.Jpeg" width="300" /><br/>| <img src="/Results_png/090113080127_0T1_time_memory_1.Jpeg" width="300" /><br/>|
## 2.4 Result Details
### CNN-based methods
* **CNN**

|Tasks|shots|Acc.(%)|Test time (s)|Trainging time (s)|Memory (KB)|
|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|T<sub>1</sub>|5|71.80|	1.183|	2.484|321|

* **CNN-FT**

|Tasks|shots|Acc.(%)|Test time (s)|Trainging time (s)|Memory (KB)|
|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|T<sub>1</sub>|5|	75.90|	3.995|	2.484|	321|
|T<sub>1</sub>|1|	48.00|	3.45|	-|		321|
|T<sub>2</sub>|5|	82.50|	5.72|	-|		225|
|T<sub>2</sub>|1|	68.00|	4.68|	-|		225|

* **CNN-MMD**

|Tasks|shots|Acc.(%)|Test time (s)|Trainging time (s)|Memory (KB)|
|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|T<sub>1</sub>|5|81.35|	1.164|	15.38|321|

### Meta-learning methods
* **MAML**

|Tasks|shots|Acc.(%)|Test time (s)|Trainging time (s)|Memory (KB)|
|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|T<sub>1</sub>|5	|95.80	|5.654	|720	|321|
|T<sub>1</sub>|1	|87.40	|4.494	|233	|321|
|T<sub>2</sub>|5	|91.95	|6.507	|312	|225|
|T<sub>2</sub>|1	|77.50	|4.455	|340	|225|

* **Reptile**

|Tasks|shots|Acc.(%)|Test time (s)|Trainging time (s)|Memory (KB)|
|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|T<sub>1</sub>|5	|94.6	|12.04	|1820	|321|
|T<sub>1</sub>|1	|bad	|-	|-	|-|
|T<sub>2</sub>|5	|91.50	|17.528	|585.6	|225|
|T<sub>2</sub>|1	|55.15	|17.59	|532	|225|

* **ProtoNet**

|Tasks|shots|Acc.(%)|Test time (s)|Trainging time (s)|Memory (KB)|
|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|T<sub>1</sub>|5	|95.30	|0.290	|41	    |160|
|T<sub>1</sub>|1	|87.69	|0.121	|24		|160||
|T<sub>2</sub>|5	|89.18	|0.161	|-		|160|
|T<sub>2</sub>|1	|77.25	|0.104	|-		|160|

* **RelationNet**

|Tasks|shots|Acc.(%)|Test time (s)|Trainging time (s)|Memory (KB)|
|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|T<sub>1</sub>|5	|92.34	|0.3472|	304	| 1339|
|T<sub>1</sub>|1	|85.65	|0.15|	102|	 1339|
|T<sub>2</sub>|5	|93.22	|0.19|	275|	 1339|
|T<sub>2</sub>|1	|77.98	|0.129|	-|	 1339|

* **MANN**

|Tasks|shots|Acc.(%)|Test time (s)|Trainging time (s)|Memory (KB)|
|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|T<sub>1</sub>|1|	88.35|	0.12|	90|4134|

**References**  
```
[1] Li, F., Chen, J., Pan, J., & Pan, T. (2020). Cross-domain learning in rotating machinery fault diagnosis under various operating conditions based on parameter transfer. Measurement Science and Technology, 31(8), 085104.  
[2] Xiao, D., Huang, Y., Zhao, L., Qin, C., Shi, H., & Liu, C. (2019). Domain adaptive motor fault diagnosis using deep transfer learning. IEEE Access, 7, 80937-80949.
[3] Finn, C., Abbeel, P., & Levine, S. (2017, July). Model-agnostic meta-learning for fast adaptation of deep networks. In International Conference on Machine Learning (pp. 1126-1135). PMLR.  
[4] Nichol, A., Achiam, J., & Schulman, J. (2018). On first-order meta-learning algorithms. arXiv preprint arXiv:1803.02999.  
[5] Santoro, A., Bartunov, S., Botvinick, M., Wierstra, D., & Lillicrap, T. (2016, June). Meta-learning with memory-augmented neural networks. In International conference on machine learning (pp. 1842-1850). PMLR.  
[6] Snell, J., Swersky, K., & Zemel, R. (2017, December). Prototypical networks for few-shot learning. In Proceedings of the 31st International Conference on Neural Information Processing Systems (pp. 4080-4090).  
[7] Sung, F., Yang, Y., Zhang, L., Xiang, T., Torr, P. H., & Hospedales, T. M. (2018). Learning to compare: Relation network for few-shot learning. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1199-1208).  
```

# 3. Our related work on meta-learning in fault diagnosis
* [Semi-supervised meta-learning networks with squeeze-and-excitation attention for few-shot fault diagnosis](https://www.sciencedirect.com/science/article/abs/pii/S0019057821001543?via%3Dihub)  
* [Similarity-based meta-learning network with adversarial domain adaptation for cross-domain fault identification](https://www.sciencedirect.com/science/article/abs/pii/S0950705121000927?via%3Dihub)  
* [Intelligent fault diagnosis of mechanical equipment under varying working condition via iterative matching network augmented with selective Signal reuse strategy](https://www.sciencedirect.com/science/article/abs/pii/S027861252030176X)  
* [Intelligent Fault Diagnosis of Satellite Communication Antenna via a Novel Meta-learning Network Combining with Attention Mechanism](https://iopscience.iop.org/article/10.1088/1742-6596/1510/1/012026)  

If you have used our codes or pretrained models in your work, please cite our following articles.
```
@article{feng2021metafault,
title = {Meta-learning as a promising approach for few-shot cross-domain fault diagnosis: Algorithms, applications, and prospects},
journal = {Knowledge-Based Systems},
pages = {107646},
year = {2021},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2021.107646},
url = {https://www.sciencedirect.com/science/article/pii/S0950705121009084},
author = {Yong Feng and Jinglong Chen and Jingsong Xie and Tianci Zhang and Haixin Lv and Tongyang Pan},
keywords = {Meta-learning, Few-shot learning, Small sample, Cross-domain, Fault diagnosis},
}
```
