import cv2
import torch
import numpy as np
from PIL import Image


def rgb2bgr(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def gray2rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def bgr2rgb():
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def to_pillow_image(image: np.ndarray):
    return Image.fromarray(image)


def batchify(out):
    """
    Take an input tensor and return it as a batch tensor with an additional dimension.

    Args:
      out: The input tensor that needs to be batchified.

    Returns:
      a PyTorch tensor. If the input tensor `out` is already a delta-prox tensor, it is returned as is.
    Otherwise, it returns a tensor with BCHW format if the input tensor has 3 dimensions with HWC
    format.
    """
    if len(out.shape) == 3 and (out.shape[2] == 1 or out.shape[2] == 3):
        out = out.permute(2, 0, 1)
    out = out.unsqueeze(0)
    return out


def to_torch_tensor(x, batch=False):
    """
    Convert a given input to a torch tensor and add a batch dimension if specified.

    Args:
      x: The input data that needs to be converted to a torch tensor.
      batch: A boolean parameter that indicates whether to add a batch dimension to the tensor or not.
    If set to True, a batch dimension will be added to the tensor. If set to False, no batch dimension
    will be added. Defaults to False

    Returns:
      a PyTorch tensor. If the input is already a dp.tensor, it returns the input as is. If the input is a
    numpy array, it converts it to a dp.tensor. If the input is neither a torch tensor nor a numpy
    array, it assumes it is a dp.tensor and returns it. If the batch parameter is True, it adds a batch
    dimension to the tensor and returns
    """
    if isinstance(x, torch.Tensor):
        out = x
    elif isinstance(x, np.ndarray):
        out = dp.tensor(x.copy())
    else:
        out = dp.tensor(x)

    if batch:
        if len(out.shape) == 3 and (out.shape[2] == 1 or out.shape[2] == 3):
            out = out.permute(2, 0, 1)
        if len(out.shape) < 4:
            out = out.unsqueeze(0)

    out.is_dp_tensor = True
    return out


def debatchify(out, squeeze):
    """
    Debatchify a tensor by squeezing and/or transposing its dimensions,
    supporting multiple tensor format transforms (BCHW -> CHW | CHW -> HWC | HWC -> HW)

    Args:
      out: The output tensor that needs to be transformed. It could be in the format of BCHW, CHW, or
    HWC.
      squeeze: A boolean parameter that determines whether to convert tensor format with C = 1 to HW
    format. If set to True, it will remove the singleton dimension and return a tensor with shape (H, W)
    instead of (H, W, 1). 

    Returns:
      the tensor `out` with simplified format.
    """
    if len(out.shape) == 4:
        out = out.squeeze(0)  # BCHW -> CHW
    if len(out.shape) == 3:
        if out.shape[0] == 3 or out.shape[0] == 1:
            out = out.transpose(1, 2, 0)  # CHW -> HWC
        if out.shape[2] == 1 and squeeze:
            out = out.squeeze(2)  # HWC -> HW
    return out


def to_ndarray(x, debatch=False, squeeze=False):
    """
    Convert a given input into a numpy array and optionally remove any batch dimensions.

    Args:
      x: The input data that needs to be converted to a numpy array.
      debatch: A boolean parameter that specifies whether to remove the batch dimension from the input
    tensor or not. If set to True, the function will call the `debatchify` function to remove the batch
    dimension. If set to False, the function will return the input tensor as is. Defaults to False
      squeeze: The `squeeze` parameter is a boolean parameter that determines whether to convert a
    tensor format with C = 1 to HW format. 

    Returns:
      a numpy array. If `debatch` is True, the output is passed through the `debatchify` function with
    `squeeze` before being returned.
    """
    if isinstance(x, torch.Tensor):
        out = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        out = x.astype('float32')
    else:
        out = np.array(x)
    if debatch:
        out = debatchify(out, squeeze)
    return out
