# coding=utf-8
import sys
import time
import torch
import pandas as pd
import functools
from collections import defaultdict

from .walk import walk_modules

class Profile(object):
    """PyTorch模型的逐层分析器，可以获取模型各层初始化、执行时间和输出数据大小"""

    def __init__(self, model, enabled=True, use_cuda=False, depth=-1):
        """
        参数：
            model：pytorch模型
            enabled：是否（True/False）启用分析
            use_cuda：是否（True/False）使用GPU
            depth：模型层嵌套深度
        """
        self._model = model
        self.enabled = enabled
        self.use_cuda = use_cuda
        self.depth = depth
        

        self.entered = False
        self.exited = False
        self.traces = ()
        self.information = defaultdict(list)

        # 输入层信息（默认输入数据size为3*224*224，float32）
        self.information["input"].extend([0, (3*224*224) * 4 / 1024 / 1024, 0.01])

    def __enter__(self):
        if not self.enabled:
            return self
        if self.entered:
            raise RuntimeError("pytorchtool profiler is not reentrant")
        self.entered = True
        self._forwards = {}  # 存储初始forwards

        # 逐层初始化分析
        self.traces = tuple(map(self._load_weight, walk_modules(self._model, depth=self.depth)))
        # 逐层修改forwards
        tuple(map(self._hook_trace, self.traces))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        # 逐层恢复初始forwards
        tuple(map(self._remove_hook_trace, self.traces))
        del self._forwards  # remove unnecessary forwards
        self.exited = True

    def __str__(self):
        return str(pd.DataFrame.from_dict(self.information, orient='index', 
            columns=['Loading Time(ms)', 'Data Size(MB)','Execute Time(ms)']))

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def _load_weight(self, trace):
        (name, module) = trace
        
        start = time.time()
        module.load_state_dict(torch.load("./model_weight/" + 
            self._model.__class__.__name__ + "/" + name + ".pth"), strict=False)
        loadingTime = (time.time() - start) * 1000
        self.information[name].append(loadingTime)
        return trace

    def _hook_trace(self, trace):
        (name, module) = trace
        _forward = module.forward
        self._forwards[name] = _forward

        @functools.wraps(_forward)
        def wrap_forward(*args, **kwargs):
            # 执行时间
            if self.use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                output = _forward(*args, **kwargs)
                end.record()
    
                # 等待执行完成
                torch.cuda.synchronize()
                
                exec_time = start.elapsed_time(end)
            else:
                start = time.time()
                output = _forward(*args, **kwargs)
                # 转换为ms
                exec_time = (time.time() - start) * 1000
            
            # 输出数据大小（MB）
            data_size = sys.getsizeof(output.storage()) / 1024 / 1024
            
            self.information[name].append(data_size)
            self.information[name].append(exec_time)
            return output

        module.forward = wrap_forward
        return trace

    def _remove_hook_trace(self, trace):
        [name, module] = trace
        module.forward = self._forwards[name]

    def printCsv(self, filePath='./parameters/default.csv'):
        """将模型分析结果写入csv文件
        参数：
            filePath：csv文件路径及文件名
        """
        df = pd.DataFrame.from_dict(self.information, orient='index', 
            columns=['Loading Time(ms)', 'Data Size(MB)','Execute Time(ms)'])
        df.to_csv(filePath)