from typing import Optional

import cv2
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


def to_pil(image: np.ndarray):
    """ convert a ndarray into a pilllow image.
        - the input ndarray can be [H,W], [H,W,3], [H,W,4], [H,W,1]
        - float32, uint8, [0,1], [0,255] are all acceptable.
        - Noted that we automatically infer the data range from max value. 
    """
    image = image.astype('float32')
    if image.max() < 1:
        image = image * 255
    image = image.astype('uint8')
    return Image.fromarray(image)


def pil_to_array(image: Image.Image, data_range=255):
    """ convert a pillow image to a float32 ndarray, range from [0,1]
        - [H,W] will be padded to [H,W,1]
    """
    image = np.array(image).astype('float32')
    image = image / data_range
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    return image


def imresize(data, size, mode='cubic'):
    """data ndarray (H,W,C) or (H,W)
    size (height, width)
    """
    height, width = size
    MODES = {
        'cubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST,
    }
    mode = MODES[mode]
    out = cv2.resize(data, dsize=(width, height), interpolation=mode)
    if len(data.shape) == 3 and data.shape[-1] == 1:
        out = out[:, :, None]
    return out


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(
    img: np.ndarray,
    mod: int,
    square: bool = False,
    min_size: Optional[int] = None,
):
    """

    Args:
        img: [H, W, C]
        mod:
        square: 是否为正方形
        min_size:

    Returns:

    """
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)

    if min_size is not None:
        assert min_size % mod == 0
        out_width = max(min_size, out_width)
        out_height = max(min_size, out_height)

    if square:
        max_size = max(out_height, out_width)
        out_height = max_size
        out_width = max_size

    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )
