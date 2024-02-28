import math
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL


def add_text(image, text):
    image = cv2.resize(image, (512, 512))
    # overlay space
    x, y, w, h = 0, 0, 250, 50

    # alpha, the 4th channel of the image
    alpha = 0.5

    overlay = image.copy()
    output = image.copy()

    # corner
    cv2.rectangle(overlay, (x, x), (x + w, y + h), (0, 0, 0), -1)

    # putText
    cv2.putText(overlay, text, (x + 50, y + int(h - 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # apply the overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


def imshow(
    *imgs: List[np.ndarray],
    maxcol: int = 3,
    gray: bool = False,
    titles: List[str] = None,
    off_axis: bool = True
):
    if len(imgs) != 1:
        plt.figure(figsize=(10, 5), dpi=300)
    row = (len(imgs) - 1) // maxcol + 1
    col = maxcol if len(imgs) >= maxcol else len(imgs)
    for idx, img in enumerate(imgs):
        if img.max() > 2:
            img = img / 255
        img = img.clip(0, 1)
        if gray:
            plt.gray()
        plt.subplot(row, col, idx + 1)
        plt.imshow(img)
        if titles is not None:
            plt.title(titles[idx])
        if off_axis:
            plt.axis('off')
    plt.show()


def image_grid(imgs):
    n = len(imgs)
    rows = int(math.sqrt(n))
    cols = math.ceil(n/rows)

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))
    return grid
