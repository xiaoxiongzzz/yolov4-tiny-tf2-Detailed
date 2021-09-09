from functools import reduce

def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)# 难点：复合函数的叠加。用自定义函数lambda叠加后面的参数序列。
    else:
        raise ValueError('Composition of empty sequence not supported.')