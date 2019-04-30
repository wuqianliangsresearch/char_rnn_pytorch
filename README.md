# RNN

### 求导过程

一般来说我们会使用一种叫混合布局的思路，即如果是向量或者矩阵对标量求导，则使用分子布局为准，如果是标量对向量或者矩阵求导，则以分母布局为准.对于向量对对向量求导，有些分歧，我的所有文章中会以分子布局的雅克比矩阵为主.


###
#### Reference
[softmax分类器+cross entropy损失函数的求导](https://www.cnblogs.com/wacc/p/5341676.html)  
[BPTT求导](http://www.cnblogs.com/wacc/p/5341670.html)  
[矩阵微分性质](https://www.cnblogs.com/pinard/p/10791506.html)  
