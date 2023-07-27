import torch
from dola.nn.benchmark import synchronize_timer

with synchronize_timer() as t:
    print('abc')
print(t())
