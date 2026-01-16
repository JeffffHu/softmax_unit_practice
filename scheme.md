# Softmax_Unit Study Scheme

整体上应该先从文献调研入手，一方面要搞清楚*softmax*函数对应的硬件*input/output*接口如何设计（搞清楚*softmax*在*Machine Learning*下面是怎么用的），另一方面要着手看一些*SFU*的文献。

> [硬件友好的高效softmax函数实现调研与分析 - 知乎](https://zhuanlan.zhihu.com/p/577554331)

> [一文详解Softmax函数 - 知乎](https://zhuanlan.zhihu.com/p/105722023)

> [电子信息学中的ML--FCN部分](./电子信息学中的机器学习FCN.pdf)

基本上, 是一个N输入N输出的单元, 其中每个输出是对应输入的指数值与所有输入指数值之和的比值, 最直观的应用就是多分类问题。