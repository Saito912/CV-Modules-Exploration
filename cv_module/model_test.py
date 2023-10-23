from thop import profile
from torch import Tensor
from torch.nn import Module
from typing import Tuple, Any, Union
from fvcore.nn import FlopCountAnalysis, flop_count_table


def model_test(model: Module, input_data: Union[Tensor, Tuple[Tensor, ...]]):
    # 创建一个随机输入张量，模拟模型的输入
    # 使用thop.profile来计算FLOPs和参数数量
    flops = FlopCountAnalysis(model, input_data)

    print(flop_count_table(flops))
    print("此处计算的flops应该×2才是我们理解的FLOPs")
    out = model(*input_data)
    return out
