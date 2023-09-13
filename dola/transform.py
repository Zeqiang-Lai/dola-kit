from typing import Optional

import cv2
import numpy as np

MODES = {
    'cubic': cv2.INTER_CUBIC,
    'nearest': cv2.INTER_NEAREST,
}


def imresize(data, size, mode='cubic'):
    """ data ndarray (H,W,C) or (H,W)
        size (height, width)
    """
    height, width = size
    mode = MODES[mode]
    out = cv2.resize(data, dsize=(width, height), interpolation=mode)
    if len(data.shape) == 3 and data.shape[-1] == 1:
        out = out[:,:,None]
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
