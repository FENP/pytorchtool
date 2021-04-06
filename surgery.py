# coding=utf-8
import sys
import torch
import logging
import functools
from collections import defaultdict, namedtuple

# name:模块名；module:模块
Trace = namedtuple("Trace", ["name", "module"])

def walk_modules(module, name="", depth=-1):
    """生成器。根据depth遍历pytorch模块，生成Trace元组"""

    child_list = list(module.named_children())
    '''
    遍历到叶子结点或depth指定的深度时返回当前模块元组；
    否则继续向下遍历
    '''
    if depth == 0 or len(child_list) == 0:
        yield Trace(name, module)
    else:
        for child in child_list:
            yield from walk_modules(child[1], child[0] if name=="" else name + "." + child[0], depth - 1)


class Surgery(object):
    """对PyTorch模型进行处理，方便进行模型切分"""

    def __init__(self, model, mode, use_cuda=False, depth=-1):
        """
        参数：
            model：初始化后的DNN模型
            mode：切分模式 0：客户端模式（执行中间层并存储输出）、2：客户端模式（根据中间层输出继续执行，得到最终结果）
        """
        self._model = model
        self._mode = mode
        self._use_cuda = use_cuda
        self._depth = depth
        
        self._layerState = None

        # 存储初始forwards
        self._forwards = {}

        # 存储中间层输出
        self._middleResult = {}

        # 逐层修改forwards
        self.traces = tuple(map(self._hook_trace, walk_modules(self._model, depth=self._depth)))

    def recover(self):
        # 逐层恢复初始forwards
        tuple(map(self._remove_hook_trace, self.traces))
        del self._forwards  # remove unnecessary forwards

    def getMiddleResult(self):
        return self._middleResult

    def setMiddleResult(self, middleResult):
        if self._mode == 2:
            self._middleResult = middleResult

    def clearMiddleResult(self):
        if self._mode == 0:
            self._middleResult.clear()

    def setLayerState(self, layerState):
        self._layerState = layerState

    def __call__(self, *args, **kwargs):
        # 针对客户端或服务端完成全部计算的清空做特殊处理
        if self._layerState['input'] == 1:
            if self._mode == 0:
                logging.info("客户端传输原始输入")
                self._middleResult['input'] = args[0]
                return torch.rand(1,1000)
            elif self._mode == 2:
                logging.info("服务端接收原始输入")
                return self._model((self._middleResult['input']))
        
        return self._model(*args, **kwargs)

    def _hook_trace(self, trace):
        [name, module] = trace
        _forward = module.forward
        self._forwards[name] = _forward

        @functools.wraps(_forward)
        def wrap_forward(*args, **kwargs):
            if self._layerState[name] != self._mode:
                # 非中间输出层直接返回原始数据
                if self._layerState[name] != 1:
                    logging.debug("skip ", name)
                    return args[0]
                # 服务端模式获取层输出并返回
                elif self._mode == 2:
                    logging.debug("middle ", name)
                    return self._middleResult[name]
            logging.debug("execute ", name)
            output = _forward(*args, **kwargs)
            
            # 客户端模型下需要存储中间层输出
            if self._mode == 0 and self._layerState[name] == 1:
                logging.debug("save ", name)
                self._middleResult[name] = output
            return output

        module.forward = wrap_forward
        return trace
    
    def _remove_hook_trace(self, trace):
        [name, module] = trace
        module.forward = self._forwards[name]