# coding=utf-8
import torch
import logging
from collections import defaultdict

class Surgery2(object):
    """分层加载模型的处理模式"""
    def __init__(self, name, dag_path, is_debug = False):
        if name in 'inception':
            self.model_name = 'Inception3'

        elif name in 'alexnet':
            self.model_name = 'AlexNet'

        elif name in 'resnet':
            self.model_name = 'ResNet'

        else:
            raise RuntimeError("Wrong model name")

        self._is_debug = is_debug

        self._edge = defaultdict(list)
        self._output = defaultdict(None)
        self._layerModule = defaultdict(None)
        # 读取dag文件
        for line in open(dag_path, 'r'):
            line = line.strip('\n')
            nameList = line.split(' ')
            name = nameList[0]
            self._output[name] = None
            self._layerModule[name] = None

            if len(nameList) == 1:
                self._endlayerName = name
            for nextLayerName in nameList[1:]:
                self._edge[nextLayerName].append(name)
    
    def loadLayer(self, layerName):
        if(self._layerModule[layerName] is None):
            if self._is_debug:
                logging.debug("初始化层: %s", layerName)
            # 加载该层
            self._layerModule[layerName] = torch.load("../pytorchtool/model_weight/" + 
                self.model_name + "/" + layerName + ".pkl")

    def inferencePart(self, middleResult):
        # 清空输出字典
        for k in self._output.keys():
            self._output[k] = None
        # 中间输出赋值
        for k, v in middleResult.items():
            self._output[k] = v
        # 获取最终结果
        return self._inferenceLayer(self._endlayerName)
    
    def _inferenceLayer(self, layerName):
        if self._output[layerName] is not None:
            return self._output[layerName]
        
        inputList = []
        for lastLayerName in self._edge[layerName]:
            inputList.append(self._inferenceLayer(lastLayerName))

        if len(inputList) == 1:
            layerInput = inputList[0]
        else:
            layerInput = inputList
        if self._is_debug:
            logging.debug("execute %s", layerName)
        self._output[layerName] = self._layerModule[layerName](layerInput)
        return self._output[layerName]
