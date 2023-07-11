import cv2


def rgb2bgr(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def gray2rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def bgr2rgb():
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
