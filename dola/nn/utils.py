from typing import Sequence, Union

import torch
import torch.nn as nn


def freeze(module: nn.Module):
    """ Freeze the model parameters """
    for _, v in module.named_parameters():
        v.requires_grad = False


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def move(data: Union[torch.Tensor, Sequence, dict], device: torch.device):
    """ Move object to device """

    if isinstance(data, torch.Tensor):
        return data.to(device).float()
    if isinstance(data, Sequence):
        return [move(d, device=device) for d in data]
    if isinstance(data, dict):
        return {k: move(v, device=device) for k, v in data.items()}
    return data


def load(path):
    if path.endswith('.safetensors'):
        from safetensors import safe_open
        tensors = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        return tensors
    return torch.load(path)


def print_parameters(net, trainable=True):
    for k, v in net.named_parameters():
        if trainable and not v.requires_grad:
            continue
        print(k, v.shape)
