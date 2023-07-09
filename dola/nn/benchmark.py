import torch


def model_size(model: torch.nn.Module, trainable: bool = False) -> float:
    """ Return the model size in M"""
    if trainable:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total = sum([param.nelement() for param in model.parameters()])
    return total / 1e6


class synchronize_timer:
    """ Synchronized timer to count the inference time of `nn.Module.forward`.
        
        Example:
        ```python
        with synchronize_timer() as t:
            run()
        print(t.time)
        ```
        
        `t.time` in ms.
    """
    
    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.end.record()
        torch.cuda.synchronize()
        self.time = self.start.elapsed_time(self.end)
