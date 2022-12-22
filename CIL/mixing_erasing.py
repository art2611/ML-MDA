import random
import torch
import numpy as np
import math
from torchvision import transforms
from TransREID.random_erasing import RandomErasing

"""Code from CIL paper + adaptations for MS-PATCH, M-REA """
def erasings(model, CIL, XPATCH, XREA, ML_MDA):
    re_erasing, random_erasing, random_erasing_IR = None, None, None

    if CIL or XPATCH != "False":
        if CIL or XPATCH in ["S-PATCH", "MS-PATCH"]:
            type_erasing = "self"
        elif XPATCH in ['M-PATCH-SS', 'M-PATCH-SD', 'M-PATCH-DD']:
            type_erasing = XPATCH
        else:
            exit(f"args.XPATCH={XPATCH} => not in ['S-PATCH', 'MS-PATCH' 'M-PATCH-SS', 'M-PATCH-SD', 'M-PATCH-DD']")

        re_erasing = mixing_erasing(probability=0.5,
                                    mean=[0.5, 0.5, 0.5],
                                    type=type_erasing,
                                    mixing_coeff=[0.5, 1.0])

    if CIL or XREA in ["S-REA", "MS-REA"] or ML_MDA:
        random_erasing = mixing_erasing(probability=0.5,
                                        mean=[0.5, 0.5, 0.5],
                                        type='soft',
                                        mixing_coeff=[0.5, 1.0])
    if XREA == "MS-REA" or ML_MDA:
        random_erasing_IR = mixing_erasing(probability=0.5,
                                           mean=[0.5, 0.5, 0.5],
                                           type='soft_IR',
                                           mixing_coeff=[0.5, 1.0])

    if model == "transreid":
        random_erasing = RandomErasing(probability=0.5, mode='pixel', max_count=1, device='cpu')

    return re_erasing, random_erasing, random_erasing_IR

class mixing_erasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels with different mixing operation.
    normal: original random erasing;
    soft: mixing ori with random pixel;
    soft_IR: mixing ori with grayscale random pixel;
    self: mixing ori with other_ori_patch;
    mmodal_sameEx_sameApp : mixing multimodal patches, same exctraction location and same applied zone
    mmodal_sameEx_diffApp : mixing multimodal patches, same exctraction location and different applied zone
    mmodal_diffEx_diffApp : mixing multimodal patches, different extraction location and different applied zone
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """
    def __init__(self,
                 probability=0.5,
                 sl=0.02,
                 sh=0.4,
                 r1=0.3,
                 mean=(0.4914, 0.4822, 0.4465),
                 mode='pixel',
                 device='cpu',
                 type='normal',
                 mixing_coeff=[1.0, 1.0]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.rand_color = False
        self.per_pixel = False
        self.mode = mode
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device
        self.type = type
        self.mixing_coeff = mixing_coeff

        self.to_grayscale = transforms.Grayscale(num_output_channels=3)

    def __call__(self, img, img2=None):
        if self.type in ["M-PATCH-SS", "M-PATCH-SD", "M-PATCH-DD"]:
            assert img2 != None

        if random.uniform(0, 1) >= self.probability:
            if img2 != None :
                return img, img2
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                # Application zone
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if self.type == 'normal':
                    m = 1.0
                else: # soft - soft_IR - self
                    m = np.float32(
                        np.random.beta(self.mixing_coeff[0],
                                       self.mixing_coeff[1]))

                if self.type == 'self':
                    # Zone patch taken
                    x2 = random.randint(0, img.size()[1] - h)
                    y2 = random.randint(0, img.size()[2] - w)
                    img[:, x1:x1 + h,
                        y1:y1 + w] = (1 - m) * img[:, x1:x1 + h, y1:y1 +
                                                   w] + m * img[:, x2:x2 + h,
                                                                y2:y2 + w]

                    return img

                elif self.type == "M-PATCH-DD":
                    for attempt in range(100):
                        target_area2 = random.uniform(self.sl, self.sh) * area
                        aspect_ratio2 = random.uniform(self.r1, 1 / self.r1)

                        h2 = int(round(math.sqrt(target_area2 * aspect_ratio2)))
                        w2 = int(round(math.sqrt(target_area2 / aspect_ratio2)))

                        if w2 < img2.size()[2] and h2 < img2.size()[1]:
                            # Application zone 1
                            x1 = random.randint(0, img.size()[1] - h2)
                            y1 = random.randint(0, img.size()[2] - w2)

                            # Application zone 2
                            x1_b = random.randint(0, img.size()[1] - h)
                            y1_b = random.randint(0, img.size()[2] - w)

                            # Zone patch taken 1
                            x2 = random.randint(0, img.size()[1] - h)
                            y2 = random.randint(0, img.size()[2] - w)

                            # Second zone patch taken 2
                            x2_b = random.randint(0, img.size()[1] - h2)
                            y2_b = random.randint(0, img.size()[2] - w2)

                            store_patch1 = img[:, x2:x2 + h, y2:y2 + w].clone()  # RGB patch
                            store_patch2 = img2[:, x2_b:x2_b + h2, y2_b:y2_b + w2].clone()  # IR patch

                            img[:, x1:x1 + h2,
                            y1:y1 + w2] = (1 - m) * img[:, x1:x1 + h2, y1:y1 +
                                                                        w2] + m * store_patch2
                            img2[:, x1_b:x1_b + h,
                            y1_b:y1_b + w] = (1 - m) * img2[:, x1_b:x1_b + h, y1_b:y1_b +
                                                                                   w] + m * store_patch1

                            return img, img2
                    return img, img2

                elif self.type == "M-PATCH-SS":
                    # Zone patch taken
                    store_patch1 = img[:, x1:x1 + h, y1:y1 + w].clone()  # RGB patch
                    store_patch2 = img2[:, x1:x1 + h, y1:y1 + w].clone()  # IR patch
                    img[:, x1:x1 + h,
                    y1:y1 + w] = (1 - m) * img[:, x1:x1 + h, y1:y1 +
                                                                w] + m * store_patch2
                    img2[:, x1:x1 + h,
                    y1:y1 + w] = (1 - m) * img2[:, x1:x1 + h, y1:y1 +
                                                                 w] + m * store_patch1
                    return img, img2

                elif self.type == "M-PATCH-SD":
                    # Second application zone
                    x1_b = random.randint(0, img.size()[1] - h)
                    y1_b = random.randint(0, img.size()[2] - w)

                    # Zone patch taken
                    x2 = random.randint(0, img.size()[1] - h)
                    y2 = random.randint(0, img.size()[2] - w)

                    store_patch1 = img[:, x2:x2 + h, y2:y2 + w].clone()  # RGB patch
                    store_patch2 = img2[:, x2:x2 + h, y2:y2 + w].clone()  # IR patch
                    img[:, x1:x1 + h,
                    y1:y1 + w] = (1 - m) * img[:, x1:x1 + h, y1:y1 +
                                                                w] + m * store_patch2
                    img2[:, x1_b:x1_b + h,
                    y1_b:y1_b + w] = (1 - m) * img2[:, x1_b:x1_b + h, y1_b:y1_b +
                                                                 w] + m * store_patch1

                    return img, img2
                else:
                    if self.mode == 'const':
                        img[0, x1:x1 + h,
                            y1:y1 + w] = (1 - m) * img[0, x1:x1 + h, y1:y1 +
                                                       w] + m * self.mean[0]
                        img[1, x1:x1 + h,
                            y1:y1 + w] = (1 - m) * img[1, x1:x1 + h, y1:y1 +
                                                       w] + m * self.mean[1]
                        img[2, x1:x1 + h,
                            y1:y1 + w] = (1 - m) * img[2, x1:x1 + h, y1:y1 +
                                                       w] + m * self.mean[2]
                    else:  # soft - soft_IR
                        if self.type == "soft":
                            img[:, x1:x1 + h, y1:y1 +
                                                 w] = (1 - m) * img[:, x1:x1 + h,
                                                                y1:y1 + w] + m * _get_pixels(self.per_pixel,
                                                                                                self.rand_color,
                                                                                                (img.size()[0], h, w),
                                                                                                dtype=img.dtype,
                                                                                                device=self.device)
                        elif self.type == "soft_IR":
                            img[:, x1:x1 + h, y1:y1 +
                                                 w] = self.to_grayscale((1 - m) * img[:, x1:x1 + h,
                                                                                  y1:y1 + w] + m * _get_pixels(
                                                                                                        self.per_pixel,
                                                                                                        self.rand_color,
                                                                                                        (img.size()[0], h, w),
                                                                                                        dtype=img.dtype,
                                                                                                        device=self.device))
                    return img
        return img


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def _get_pixels(per_pixel,
                rand_color,
                patch_size,
                dtype=torch.float32,
                device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype,
                           device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)
