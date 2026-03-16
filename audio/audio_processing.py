from torch import log
from torch import exp
from torch import clamp

def dynamic_range_compression(x, C=1, clip_value=1e-5):
    # C is compression factor
    return log(clamp(x, min=clip_value)* C)

def dynamic_range_decompression(x, C=1):
    # C <- used same value as compression
    return exp(x) / C