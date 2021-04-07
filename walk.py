def walk_modules(module, name="", depth=-1):
    """生成器。根据depth遍历pytorch模块，生成Trace元组"""
    
    child_list = list(module.named_children())
    '''
    遍历到叶子结点或depth指定的深度时返回当前模块元组；
    否则继续向下遍历
    '''
    if depth == 0 or len(child_list) == 0:
        yield (name, module)
    else:
        for child in child_list:
            yield from walk_modules(child[1], child[0] if name=="" else name + "." + child[0], depth - 1)