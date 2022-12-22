import torch
import random
from typing import Any
from types import FunctionType
from corruption.corruptions import *

corruption_function_RGB = [gaussian_noise, shot_noise, impulse_noise, defocus_blur,
                           glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast,
                           elastic_transform, pixelate, jpeg_compression, speckle_noise,
                           gaussian_blur, spatter, saturate, rain]  # 20

# All the same for IR except that brightness is used to simulate saturate
# (so saturate function does not appear the list but brightess appears)
corruption_function_IR = [gaussian_noise, shot_noise, impulse_noise, defocus_blur,
                          glass_blur, motion_blur, zoom_blur, snow, frost, fog, contrast,
                          elastic_transform, pixelate, jpeg_compression, speckle_noise,
                          gaussian_blur, spatter, brightness, rain, none]  # 20

class Masking(object):
    def __init__(self, ratio=0.8):
        self.type = type

        self.ratio = ratio  # Probability that one pair of images contain a masked img
        self.rng = random.Random()  # local random seed

    def __call__(self, img_RGB, img_IR):

        img_list = [img_RGB, img_IR]
        modality_selection = self.rng.choice([[0, 1], [1, 0]])  # Determine from which modality we select the corruption first

        if self.rng.random() < 0.5:
            if self.rng.random() < self.ratio/2 :
                img_list[modality_selection[0]] = Image.new(mode="RGB", size=(144, 288), color=(255, 255, 255))
                return img_list # Two masked images never happened, we avoid having two blanked images.

        if self.rng.random() < 0.5:
            if self.rng.random() < self.ratio/2 :
                img_list[modality_selection[1]] = Image.new(mode="RGB", size=(144, 288), color=(255, 255, 255))

        return img_list

# Specific compose class to handle multimodal data transformations
class Compose:
    def __init__(self, transforms, CIL, XPATCH, scenario_eval="normal"):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            _log_api_usage_once(self)
        self.transforms = transforms
        self.XPATCH = XPATCH
        self.CIL = CIL
        self.scenario_eval = scenario_eval
        self.first_roll = True
    def __call__(self, img_RGB, img_IR=None):
        for t in self.transforms:
            if "mixing_erasing" in str(t):
                if t.type == "soft":
                    img_RGB, img_IR = t(img_RGB), img_IR
                if t.type == "soft_IR":
                    img_RGB, img_IR = img_RGB, t(img_IR)
                if t.type == "self" and (self.XPATCH == "S-PATCH" or self.CIL):
                    img_RGB, img_IR = t(img_RGB), img_IR
                if t.type == "self" and self.XPATCH == "MS-PATCH":
                    img_RGB, img_IR = t(img_RGB), t(img_IR)
            elif self.scenario_eval != "normal" and "Corruption_transform" in str(t):
                if self.scenario_eval == "C":  # Apply corrupt on RGB only
                    img_RGB, img_IR = t(img_RGB, modality="RGB"), img_IR
                elif self.scenario_eval == "C*": # Apply corrupt on both RGB and IR
                    img_RGB, img_IR = t(img_RGB, modality="RGB"), t(img_IR, modality="IR")
            else :
                try : # Handle most of the transformations like toTensor, horizontal flips, crops etc..
                    img_RGB = t(img_RGB)
                    img_IR = t(img_IR)
                except:  # Handle and multimodal patch mixing
                    img_RGB, img_IR = t(img_RGB, img_IR)

        return img_RGB, img_IR

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

# Adapted from https://github.com/MinghuiChen43/CIL-ReID/blob/main/datasets/make_dataloader.py
class corruption_transform(object):
    def __init__(self, level=0, type='all'):
        self.level = level
        self.type = type

        self.corruption_function_RGB = corruption_function_RGB  # if modality == "RGB" else corruption_function_IR
        self.corruption_function_IR = corruption_function_IR  # if modality == "RGB" else corruption_function_IR

        self.rng = random  # local random seed

    def __call__(self, img, modality="RGB"):
        corruption_function = self.corruption_function_RGB if modality == "RGB" else self.corruption_function_IR
        if self.level > 0 and self.level < 6:
            level_idx = self.level
        else:
            level_idx = self.rng.choice(range(1, 6))
        if self.type == 'all':
            corrupt_func = self.rng.choice(corruption_function)
        else:
            func_name_list = [f.__name__ for f in corruption_function]
            corrupt_idx = func_name_list.index(self.type)
            corrupt_func = corruption_function[corrupt_idx]

        c_img = corrupt_func(img.copy(), severity=level_idx, modality=modality)
        img = Image.fromarray(np.uint8(c_img))
        return img

def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;

    Args:
        obj (class instance or method): an object to extract info from.
    """
    if not obj.__module__.startswith("torchvision"):
        return
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{obj.__module__}.{name}")
