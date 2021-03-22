# pytorch-tool
DNN_Partition辅助工具，用于对pytorch模型进行简单的性能分析

#### 使用步骤

1. 对一个初始化后的模型（已加载权重文件）使用`save_model`函数分别将各层权重保存到**“./model_weight/”**
2. 对一个未初始化的模型，使用`with Profile(model) as p`，然后执行模型推理。模型分析被保存在`self.information`中，使用`p.printCsv`可以以csv文件格式输出。

**添加了示例（examp1.py classes.py）**

