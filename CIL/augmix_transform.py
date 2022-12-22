import numpy as np
from PIL import Image
from .augmix.augmix import augmix


# input PIL, output PIL, put on the top of all transforms
class augmix_transform(object):
    def __init__(self, level=0, width=3, depth=-1, alpha=1.):
        self.level = level
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def __call__(self, img):
        img = augmix(np.asarray(img) / 255)
        img = np.clip(img * 255., 0, 255).astype(np.uint8)
        img = Image.fromarray(img)
        return img