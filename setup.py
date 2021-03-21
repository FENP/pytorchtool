# coding=utf-8
import os
import torch

path = "./model_weight"
dir_name = "default"

def save_weight_by_layer(module, name="", depth=-1):
    """根据depth遍历pytorch模块，保存模块的权重文件"""

    child_list = list(module.named_children())
    '''
    遍历到叶子结点或depth指定的深度时保存权重；
    否则继续向下遍历
    '''
    if depth == 0 or len(child_list) == 0:
        torch.save(module, os.path.join(path, dir_name, name + ".pth"))
    else:
        for child in child_list:
            save_weight_by_layer(child[1], child[0] if name=="" else name + "." + child[0], depth - 1)

def save_model(model, depth=-1):
    global dir_name
    dir_name = model.__class__.__name__
    save_weight_by_layer(model, name="", depth=depth)