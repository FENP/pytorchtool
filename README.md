# pytorchtool
DNN_Partition辅助工具，用于对pytorch模型进行简单的性能分析

#### 使用步骤

1. 对一个初始化后的模型（已加载权重文件）使用`save_model`函数分别将各层权重保存到**“./model_weight/”**
2. 对一个未初始化的模型，使用`with Profile(model) as p`，然后执行模型推理。模型分析被保存在`self.information`中，使用`p.printCsv`可以以csv文件格式输出。

**添加了示例（example.py classes.py）** *已验证*


```shell
# AlexNet
mkdir alexnet
cd ./alexnet
wget https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
```

```shell
# inceptionV3
mkdir inception_v3
cd ./inception_v3
wget https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth
```

## 性能分析

### 1. 保存各层权重文件

`doPrepare=True`，执行`python example.py`

### 2. 获取各层性能参数

`doPrepare=False `, `doProf=True`，执行`python example.py`