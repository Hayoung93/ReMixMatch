import torch
import torch.nn as nn
from torch.distributions import categorical
from torchvision import transforms as ttf


class CTAugment(nn.Module):
    def __init__(self, depth=2, th=0.85, decay=0.99):
        """
        depth: number of selected transformations for each input image
        """
        super().__init__()
        # TODO: cutout, invert, rescale, smooth
        self.aug_list = [Rotate, ShearX, ShearY, TranslateX, TranslateY, AutoContrast,
                        Sharpness, Identity, Contrast, Color, Brightness, Equalize, 
                        Solarize, Posterize]
        self.depth = depth
        self.th = th
        self.decay = decay
        self.bins = torch.ones((len(self.aug_list), 17))
    
    def update(self, preds, labels, aug_index, bin_index):
        # compute update weight and apply to bins
        w = 1 -  1 / (2 * len(labels)) * (preds - labels).abs().sum()
        for ai, bi in zip(aug_index, bin_index):
            if ai.shape[0] > 1:
                for aii, bii in zip(ai, bi):
                    self.bins[aii][bii] = self.decay * self.bins[aii][bii] + (1 - self.decay) * w
            else:
                self.bins[ai][bi] = self.decay * self.bins[ai][bi] + (1 - self.decay) * w
    
    def aug_batch(self, batch):
        aug_index = []
        bin_index = []
        with torch.no_grad():
            for image in batch:
                augs, aug_ind, bin_ind = self.sample()
                for aug in augs:
                    image = aug(image)
                aug_index.append(aug_ind)
                bin_index.append(bin_ind)
        return batch, aug_index, bin_index

    def sample(self):
        # sample augmentations from categorical distribution according to bins
        augs = []
        aug_index = torch.randint(low=0, high=len(self.aug_list), size=(self.depth,))
        strengths = []
        for ai in aug_index:
            probs = self.bins[ai]
            probs[probs<=self.th] = 0
            probs = probs / probs.sum()
            bin_sampler = categorical.Categorical(probs)
            strength = bin_sampler.sample()
            strengths.append(strength)
            augs.append(self.aug_list[ai](strength))
        return augs, aug_index, strengths


class Rotate(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.angle = 359 / 16 * self.M
    
    def forward(self, img):
        return ttf.functional.rotate(img, self.angle.item())


class ShearX(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.angle = 359 / 16 * self.M - 180
    
    def forward(self, img):
        return ttf.functional.affine(img, 0, [0, 0], 1, [self.angle, 0])


class ShearY(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.angle = 359 / 16 * self.M - 180
    
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
        return ttf.functional.affine(img, 0, [(max_size - 1) / 16 * self.M, 0], 1, [0, 0])


class TranslateY(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
    
    def forward(self, img):
        try:
            max_size = img.size()[1]
        except TypeError:
            max_size = img.size()[1]
        return ttf.functional.affine(img, 0, [0, (max_size - 1) / 16 * self.M], 1, [0, 0])


class AutoContrast(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M / 16 * 10
    
    def forward(self, img):
        return ttf.functional.autocontrast(img)


class Sharpness(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M / 16 * 10
    
    def forward(self, img):
        return ttf.functional.adjust_sharpness(img, self.M / 5.)


class Identity(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M / 16 * 10
    
    def forward(self, img):
        return img


class Contrast(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M / 16 * 10
    
    def forward(self, img):
        return ttf.functional.adjust_contrast(img, self.M / 5.)


class Color(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M / 16 * 10
    
    def forward(self, img):
        return ttf.functional.adjust_saturation(img, self.M / 5.)


class Brightness(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M / 16 * 10
    
    def forward(self, img):
        return ttf.functional.adjust_brightness(img, self.M / 5.)


class Equalize(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M / 16 * 10
    
    def forward(self, img):
        return ttf.functional.equalize(img)


class Solarize(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M / 16 * 10
    
    def forward(self, img):
        return ttf.functional.solarize(img, (10 - self.M) * 25.5)


class Posterize(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M / 16 * 10
    
    def forward(self, img):
        return ttf.functional.posterize(img, ((10 - self.M) / 10 * 8).round())
