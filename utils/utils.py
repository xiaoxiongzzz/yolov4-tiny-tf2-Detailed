from functools import reduce
from PIL import Image

def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)# 难点：复合函数的叠加。用自定义函数lambda叠加后面的参数序列。
    else:
        raise ValueError('Composition of empty sequence not supported.')

# 这个功能可以多百度，他是采取找到最小的那个缩放比后进行缩放。以保证图片不失真进入并缩放进入模型
def letterbox_image(image, size):
    iw, ih = image.size
    w, h   = size
    scale = min (w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    # BICUBIC更清晰，ANTIALIAS插值算法也可尝试，速度较快！
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128)) # 128,128,128是RGB的灰度值
    new_image.paste(image, ((w-nw)//2, (h-nh)//2)) # 贴灰条，将缩放后的图贴在会灰图上
    return new_image

