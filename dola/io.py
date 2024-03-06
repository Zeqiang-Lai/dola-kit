import os
import glob
import tqdm
import cv2
import json
import varname
from PIL import Image

import torch
import numpy as np

from rich.console import Console


def lo(*xs, verbose=0):

    console = Console()

    def _lo(x, name):

        if isinstance(x, np.ndarray):
            # general stats
            text = ""
            text += f"[orange1]Array {name}[/orange1] {x.shape} {x.dtype}"
            if x.size > 0:
                text += f" ∈ [{x.min()}, {x.max()}]"
            if verbose >= 1:
                text += f" μ = {x.mean()} σ = {x.std()}"
            # detect abnormal values
            if np.isnan(x).any():
                text += "[red] NaN![/red]"
            if np.isinf(x).any():
                text += "[red] Inf![/red]"
            console.print(text)

            # show values if shape is small or verbose is high
            if x.size < 50 or verbose >= 2:
                # np.set_printoptions(precision=4)
                print(x)

        elif torch.is_tensor(x):
            # general stats
            text = ""
            text += f"[orange1]Tensor {name}[/orange1] {x.shape} {x.dtype} {x.device}"
            if x.numel() > 0:
                text += f"∈ [{x.min().item()}, {x.max().item()}]"
            if verbose >= 1:
                text += f" μ = {x.mean().item()} σ = {x.std().item()}"
            # detect abnormal values
            if torch.isnan(x).any():
                text += "[red] NaN![/red]"
            if torch.isinf(x).any():
                text += "[red] Inf![/red]"
            console.print(text)

            # show values if shape is small or verbose is high
            if x.numel() < 50 or verbose >= 2:
                # np.set_printoptions(precision=4)
                print(x)

        else:  # other type, just print them
            console.print(f"[orange1]{type(x)} {name}[/orange1] {x}")

    # inspect names
    for i, x in enumerate(xs):
        _lo(x, varname.argname(f"xs[{i}]"))


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(x, path, **kwargs):
    with open(path, "w") as f:
        json.dump(x, f, **kwargs)


def load_image(path, mode="float", order="RGB"):
    from .image import make_transparent_background_white
    if mode == "pil":
        image = Image.open(path)
        image = np.array(image)
        image = make_transparent_background_white(image)
        return Image.fromarray(image).convert(order)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # cvtColor
    if order == "RGB":
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = make_transparent_background_white(img)

    # mode
    if "float" in mode:
        return img.astype(np.float32) / 255
    elif "tensor" in mode:
        return torch.from_numpy(img.astype(np.float32) / 255)
    else:  # uint8
        return img


def save_image(img, path, order="RGB"):

    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()

    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)

    # cvtColor
    if order == "RGB":
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    dir_path = os.path.dirname(path)
    if dir_path != '' and not os.path.exists(dir_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
