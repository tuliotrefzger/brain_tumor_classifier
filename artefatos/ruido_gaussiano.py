from skimage.util import random_noise
import numpy as np


def to_0255(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    return img.astype(np.uint8)


# deg_level de 1 a 10
def generate_ruido_gaussiano(img, deg_level):
    # Noise Levels
    sigmas = np.linspace(1, 10, 10) / 40
    return to_0255(random_noise(img, var=sigmas[deg_level - 1] ** 2))
