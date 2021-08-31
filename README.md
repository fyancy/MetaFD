# Meta-Learning-in-Fault-Diagnosis
The source codes for Meta-learning for few-shot cross-domain fault diagnosis.

# Instructions
1. To run all models, the requirements of your python environmrnt are as: 1) pytorch 1.8.1+cu102; 2) tensorflow-gpu 2.4.0. Note that only `MANN` is implemented by tensorflow, all other methods are achieved by pytorch. Thus, with pytorch only, you can observe the performance of most methods on CWRU dataset.
2. Some packages you have to install: 1) tensorflow_addons (for AdamW in tensorflow. Not really necessary) 2) learn2learn. The latter is an advanced API to achieve meta-learning methods, which is definitely compatible with pytorch. If you have problems when installing learn2learn, such as 'Microsoft Visual C++ 14.0 is required.', please refers to https://zhuanlan.zhihu.com/p/165008313
3. For methods given, we present the performance as follows.

