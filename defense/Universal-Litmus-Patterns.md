# code for 'Universal-Litmus-Patterns'
文章《Universal Litmus Patterns》的代码，做了一些改动:
1. 原来的程序只能使用结构完全相同的清洁或后门模型来训练后门检测器的参数，改动后可以使用任何不同结构的CNN网络作为检测器的训练和测试数据。
2. 原始程序只是为了论文所需要的实验所设计，要想实现后门检测，使用起来并不方便，因此对程序进行了封装，封装之后按照库函数的形式调用即可。

检测器的训练和使用方法：
```python
from ULP import ULP
ulp = ULP(N=10, W=32, H=32, nofclasses=10)
ulp.train(train_datas=train_datas, train_labels=labels_train, epochs=100)
ulp.save('my_ulp.pkl')
ulp.load('my_ulp.pkl')
print(ulp.predict(data))
```
使用内置参数快速进行检测的方法：
```python
from ULP import predict_fast
print(predict_fast(data, x_shape=[32,32])))
```

关于此代码的下一步计划：学习并整理几种典型的后门攻击方法的程序，使用不同的后门攻击方法生成不同结构的后门网络，用来训练和测试本文提到的石蕊模式的后门检测方法的有效性。

计划想要使用之前训练好的检测器去检测一下融合后门攻击生成的后门模型，但是发现了一个问题。检测器是将模型对特定输入样本的的输出作为检测器的输入，而模型输出的长度和检测类别相同，因此检测器只能检测特定类别数量的模型，如此一来该方法的实用性就很低了，目前还没有想到好的解决办法。

