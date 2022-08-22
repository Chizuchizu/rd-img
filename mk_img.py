import numpy as np
import cv2

def make_random_img(height, width):
    return np.random.randint(0, 256, (height, width, 3))


def make_white_img(height, width):
    return np.zeros((height, width, 3), dtype=np.uint8) + 255

