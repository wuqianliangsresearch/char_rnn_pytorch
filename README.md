# RNN

[//]: # (Image References)
[image1]: ./data/vanillarn.bmp

### 典型网络结构
![vanilla rnn][image1]
### 求导过程
对于分子布局(numerator layout),求导结果中分子保持原始形式,分母为转置形式。
对于分母布局(denominator layout),求导结果中分子为转置形式,分母保持原始形式。
本文中的求导布局都是分母布局。



###
#### Reference
[softmax分类器+cross entropy损失函数的求导](https://www.cnblogs.com/wacc/p/5341676.html)  
[BPTT求导](http://www.cnblogs.com/wacc/p/5341670.html)  
[矩阵微分性质](https://www.cnblogs.com/pinard/p/10791506.html)  
