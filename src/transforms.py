import torchvision.transforms.functional as F
import random

class RandTransforms:
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        x = self.rand_brightness(x)
        x = self.rand_contrast(x)
        x = self.rand_saturation(x)
        x = self.rand_hue(x)
        x = self.rand_horizantal_flip(x)
        x = self.rand_translate(x)
        x = self.rand_rotate(x)
        return x

    def apply(self):
        rand = random.uniform(0.0, 1.0)
        apply = rand < self.p
        #print(apply)
        return apply


    def rand_brightness(self, x):
        if self.apply():
            brightness_factor = random.uniform(0.0, 2.0)
            x = F.adjust_brightness(x, brightness_factor)
        return x

    def rand_contrast(self, x):
        if self.apply():
            contrast_factor = random.uniform(0.0, 2.0)
            x = F.adjust_contrast(x, contrast_factor)
        return x

    def rand_saturation(self, x):
        if self.apply():
            saturation_factor = random.uniform(0.0, 2.0)
            x = F.adjust_saturation(x, saturation_factor)
        return x

    def rand_hue(self, x):
        if self.apply():
            hue_factor = random.uniform(-0.5, 0.5)
            x = F.adjust_hue(x, hue_factor)
        return x

    def rand_horizantal_flip(self, x):
        if self.apply():
            x = F.hflip(x)
        return x

    def rand_translate(self, x):
        if self.apply():
            translate_x = random.uniform(-0.125, 0.125)
            translate_y = random.uniform(-0.125, 0.125)
            x = F.affine(x, translate=[translate_x, translate_y], interpolation=F.InterpolationMode.BILINEAR, angle=0, scale=1, shear=0)
        return x

    def rand_rotate(self, x):
        if self.apply():
            angle = random.uniform(-90.0, 90.0)
            x = F.rotate(x, angle=angle, interpolation=F.InterpolationMode.BILINEAR)
        return x

