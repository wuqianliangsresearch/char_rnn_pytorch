# RNN

[//]: # (Image References)
[image1]: ./data/vanillarn.bmp
[image2]: ./data/signal.bmp
[image3]: ./data/formula.bmp
[step1]: ./data/step1.bmp
[step2]: ./data/step2.bmp
[step3]: ./data/step3.bmp
[y]: ./data/y.bmp
[softmax]: ./data/softmax.bmp
[CE]: ./data/CE.bmp
[Y_HAT_Z]: ./data/1.bmp
[CE_Y_HAT]: ./data/2.bmp
[CE_Z]: ./data/CE_DI.bmp

### 典型网络结构
![vanilla rnn][image1]
### 求导过程

对于分子布局(numerator layout),求导结果中分子保持原始形式,分母为转置形式。
对于分母布局(denominator layout),求导结果中分子为转置形式,分母保持原始形式。
本文中的符合函数的求导布局都是分母布局。标量对标量的求导 还是默认的分子布局。

#### 主要符号和公式

![符号表][image2]

主要公式：

![主要公式][image3]

#### step1

![1][step1]

里面有一个交叉熵对Zt求导的：

y的定义：

![y的定义][y]

softmax的定义：

![softmax的定义][softmax]

CE的定义：

![CE的定义][CE]

![Y_HAT_Z][Y_HAT_Z]

![CE_Y_HAT][CE_Y_HAT]

![CE_Z][CE_Z]

#### step2

![2][step2]

#### step3

![3][step3]

**
#### Reference
[softmax分类器+cross entropy损失函数的求导](https://www.cnblogs.com/wacc/p/5341676.html)  
[BPTT求导](http://www.cnblogs.com/wacc/p/5341670.html)  
[矩阵微分性质](https://www.cnblogs.com/pinard/p/10791506.html)  
