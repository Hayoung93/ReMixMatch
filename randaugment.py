import torch
import torch.nn as nn
from torchvision import transforms as ttf


class RandAugment(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        """
        rotate
        shear x
        shear y
        translate y
        translate x
        autoContrast
        sharpness
        identity
        contrast
        color
        brightness
        eqaulize
        solarize
        posterize
        """
        self.N = N
        self.M = M

        self.aug_list = [Rotate, ShearX, ShearY, TranslateX, TranslateY, AutoContrast,
                        Sharpness, Identity, Contrast, Color, Brightness, Equalize, 
                        Solarize, Posterize]

    def forward(self, img):
        self.aug_index = torch.randperm(len(self.aug_list))[:self.N]
        self.augmentations = nn.ModuleList([])
        for aug_id in self.aug_index:
            self.augmentations.append(self.aug_list[aug_id](self.M))
        self.augmentations = nn.Sequential(*self.augmentations)
        return self.augmentations(img)


class Rotate(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.angle = 359 / 10 * self.M
    
    def forward(self, img):
        return ttf.functional.rotate(img, self.angle)


class ShearX(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.angle = 359 / 10 * self.M - 180
    
    def forward(self, img):
        return ttf.functional.affine(img, 0, [0, 0], 1, [self.angle, 0])


class ShearY(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.angle = 359 / 10 * self.M - 180
    
    def forward(self, img):
        return ttf.functional.affine(img, 0, [0, 0], 1, [0, self.angle])


class TranslateX(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
    
    def forward(self, img):
        try:
            max_size = img.size()[0]
        except TypeError:
            max_size = img.size()[0]
        return ttf.functional.affine(img, 0, [(max_size - 1) / 10 * self.M, 0], 1, [0, 0])


class TranslateY(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
    
    def forward(self, img):
        try:
            max_size = img.size()[1]
        except TypeError:
            max_size = img.size()[1]
        return ttf.functional.affine(img, 0, [0, (max_size - 1) / 10 * self.M], 1, [0, 0])


class AutoContrast(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
    
    def forward(self, img):
        return ttf.functional.autocontrast(img)


class Sharpness(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
    
    def forward(self, img):
        return ttf.functional.adjust_sharpness(img, self.M / 5.)


class Identity(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
    
    def forward(self, img):
        return img


class Contrast(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
    
    def forward(self, img):
        return ttf.functional.adjust_contrast(img, self.M / 5.)


class Color(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
    
    def forward(self, img):
        return ttf.functional.adjust_saturation(img, self.M / 5.)


class Brightness(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
    
    def forward(self, img):
        return ttf.functional.adjust_brightness(img, self.M / 5.)


class Equalize(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
    
    def forward(self, img):
        return ttf.functional.equalize(img)


class Solarize(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
    
    def forward(self, img):
        return ttf.functional.solarize(img, (10 - self.M) * 25.5)


class Posterize(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
    
    def forward(self, img):
        return ttf.functional.posterize(img, round((10 - self.M) / 10 * 8))