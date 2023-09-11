from thop import profile
from torch.nn import Module
from typing import Tuple, Any


def model_test(model: Module, input_data: Tuple[Any]):
    # 创建一个随机输入张量，模拟模型的输入
    # 使用thop.profile来计算FLOPs和参数数量
    flops, params = profile(model, inputs=input_data)
    print(f"FLOPs: {flops / 1e9} G FLOPs")  # 以十亿FLOPs为单位打印
    print(f"参数数量: {params / 1e6} 百万")
    out = model(*input_data)
    return out
