import cv2
from typing import List

import numpy as np
import matplotlib.pyplot as plt


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


def imshow(*imgs: List[np.ndarray],
           maxcol: int = 3,
           gray: bool = False,
           titles: List[str] = None,
           off_axis: bool = True) -> None:
    """
    Display one or more images in a grid with customizable parameters such as 
    maximum number of columns, grayscale, and titles.

    Args:
      imgs (List[np.ndarray]): a list of images.
      maxcol (int): The maximum number of columns to display the images in. If there are more images than
        maxcol, they will be displayed in multiple rows. The default value is 3, defaults to 3 (optional)
      gray (bool): A boolean parameter that determines whether the image(s) should be displayed in
        grayscale or in color. If set to True, the images will be displayed in grayscale. If set to False,
        the images will be displayed in color, defaults to False (optional)
      titles (List[str]): titles is a list of strings that contains the titles for each image being displayed. If titles is None, then no titles will be displayed
      off_axis (bool): whether to remove axis in the images.
    """
    if len(imgs) != 1:
        plt.figure(figsize=(10, 5), dpi=300)
    row = (len(imgs) - 1) // maxcol + 1
    col = maxcol if len(imgs) >= maxcol else len(imgs)
    for idx, img in enumerate(imgs):
        if img.max() > 2: img = img / 255
        img = img.clip(0, 1)
        if gray: plt.gray()
        plt.subplot(row, col, idx + 1)
        plt.imshow(img)
        if titles is not None: plt.title(titles[idx])
        if off_axis: plt.axis('off')
    plt.show()
