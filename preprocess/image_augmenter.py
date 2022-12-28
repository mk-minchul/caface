import random
import numpy as np
from torchvision.transforms import functional as F
import cv2
from torchvision import transforms
from PIL import Image
import imgaug.augmenters as iaa
import torch
import os


class MultipleAugmenter():
    def __init__(self):

        self.augmenter = Augmenter()

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def make_aug_samples(self, path, num_aug=16):

        assert os.path.isfile(path), 'no image file found'
        img = cv2.imread(path)
        sample = Image.fromarray(img)

        aug_samples = self.augmenter.sequential_augment(sample, n=num_aug)
        aug_tensors = [self.transform(a_sample) for a_sample in aug_samples]

        return torch.stack(aug_tensors, dim=0)


class Augmenter():
    def __init__(self):

        self.geo_aug = GeometricAugmenter()
        self.photo_aug = PhotometricAugmenter()
        self.blur_aug = BlurAugmenter()
        self.no_aug = NoAugment()

    def strong_augment(self, sample):
        sample = self.geo_aug.augment(sample)
        sample = self.photo_aug.augment(sample)
        sample = self.blur_aug.augment(sample)
        return sample

    def weak_augment(self, sample):
        aug = self.get_random_augmenter()
        sample = aug.augment(sample)
        return sample

    def augment(self, sample):
        if random.random() < 0.5:
            return self.strong_augment(sample)
        else:
            return self.weak_augment(sample)

    def sequential_augment(self, sample, n=16):
        all_augs = []
        strong_aug = self.strong_augment(sample)
        all_augs.append(strong_aug)
        for _ in range(n-1):
            weak_aug = self.weak_augment(strong_aug)
            all_augs.append(weak_aug)
        return all_augs

    def get_random_augmenter(self):
        aug_method = np.random.choice(['photo', 'blur', 'geo'])
        if aug_method == 'photo':
            augmenter = self.photo_aug
        elif aug_method == 'blur':
            augmenter = self.blur_aug
        elif aug_method == 'geo':
            augmenter = self.geo_aug
        else:
            raise ValueError('not a correct aug')
        return augmenter


class NoAugment():
    def __init__(self):
        pass
    def sample_param(self, sample):
        return None
    def augment(self, sample, param):
        return sample

class GeometricAugmenter():

    def __init__(self, crop_prob=0.5):
        self.crop_aug = CropAugmenter()
        self.affine_aug = AffineAugmenter()
        self.rotate_aug = RotateAugmenter()
        self.crop_prob = crop_prob

    def get_random_augmenter(self):
        aug_method = np.random.choice(['crop', 'affine', 'rotate'])
        if aug_method == 'crop':
            augmenter = self.crop_aug
        elif aug_method == 'affine':
            augmenter = self.affine_aug
        elif aug_method == 'rotate':
            augmenter = self.rotate_aug
        else:
            raise ValueError('not a correct aug')
        return augmenter

    def sample_param(self, sample):
        aug = self.get_random_augmenter()
        param = aug.sample_param(sample)
        return aug, param

    def augment(self, sample, param=None):
        if param is None:
            aug, param = self.sample_param(sample)
        else:
            aug, param = param
        return aug.augment(sample, param)

class CropAugmenter():
    def __init__(self):
        self.random_resized_crop = transforms.RandomResizedCrop(size=(112, 112),
                                                                scale=(0.2, 1.0),
                                                                ratio=(0.75, 1.3333333333333333))

    def sample_param(self, sample):
        i, j, h, w = self.random_resized_crop.get_params(sample,
                                                         self.random_resized_crop.scale,
                                                         self.random_resized_crop.ratio)
        shift_x = 0
        shift_y = 0
        return i, j, h, w, shift_x, shift_y

    def augment(self, sample, param=None):
        if param is None:
            param = self.sample_param(sample)
        new = np.zeros_like(np.array(sample))

        i, j, h, w, shift_x, shift_y = param
        cropped = F.crop(sample, i, j, h, w)
        new[i:i+h, j:j+w, :] = np.array(cropped)
        sample = Image.fromarray(new.astype(np.uint8))

        return sample

class AffineAugmenter():
    def __init__(self):
        pass

    def sample_param(self, sample):
        zoom = np.random.uniform(0.5, 1.5)
        trans_x = np.random.uniform(-0.2, 0.2)
        trans_y = np.random.uniform(-0.2, 0.2)
        return zoom, trans_x, trans_y

    def augment(self, sample, param=None):
        if param is None:
            param = self.sample_param(sample)
        zoom, trans_x, trans_y = param
        affine = iaa.Affine(scale=zoom, translate_percent={'x':trans_x, 'y':trans_y})
        affined_img = affine(image=np.array(sample))
        sample = Image.fromarray(affined_img.astype(np.uint8))

        return sample

class RotateAugmenter():
    def __init__(self):
        pass

    def sample_param(self, sample):
        angle = np.random.uniform(-25, 25)
        return angle

    def augment(self, sample, param=None):
        if param is None:
            param = self.sample_param(sample)
        angle = param
        aug = iaa.Affine(rotate=angle)
        rotated_img = aug(image=np.array(sample))
        sample = Image.fromarray(rotated_img.astype(np.uint8))
        return sample


class PhotometricAugmenter():

    def __init__(self,):
        self.photometric = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)

    def sample_param(self, sample):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.photometric.get_params(self.photometric.brightness, self.photometric.contrast,
                                        self.photometric.saturation, self.photometric.hue)
        return fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor

    def augment(self, sample, param=None):
        if param is None:
            param = self.sample_param(sample)
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = param

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                sample = F.adjust_brightness(sample, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                sample = F.adjust_contrast(sample, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                sample = F.adjust_saturation(sample, saturation_factor)

        return sample


class BlurAugmenter():

    def __init__(self, ):
        pass

    def sample_param(self, sample):
        blur_method = np.random.choice(['avg', 'gaussian', 'motion', 'resize'])
        if blur_method == 'avg':
            k = np.random.randint(1, 10)
            param = [blur_method, k]
        elif blur_method == 'gaussian':
            sigma = np.random.random() * 4
            param = [blur_method, sigma]
        elif blur_method == 'motion':
            k = np.random.randint(5, 20)
            angle = np.random.randint(-45, 45)
            direction = np.random.random() * 2 - 1
            param = [blur_method, k, angle, direction]
        elif blur_method == 'resize':
            side_ratio = np.random.uniform(0.2, 1.0)
            interpolation = np.random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
            param = [blur_method, side_ratio, interpolation]
        else:
            raise ValueError('not a correct blur')

        return param

    def augment(self, sample, param=None):
        if param is None:
            param = self.sample_param(sample)
        blur_method = param[0]
        if blur_method == 'avg':
            blur_method, k = param
            avg_blur = iaa.AverageBlur(k=k) # max 10
            blurred = avg_blur(image=np.array(sample))
        elif blur_method == 'gaussian':
            blur_method, sigma = param
            gaussian_blur = iaa.GaussianBlur(sigma=sigma) # 4 is max
            blurred = gaussian_blur(image=np.array(sample))
        elif blur_method == 'motion':
            blur_method, k, angle, direction = param
            motion_blur = iaa.MotionBlur(k=k, angle=angle, direction=direction)  # k 20 max angle:-45 45, dir:-1 1
            blurred = motion_blur(image=np.array(sample))
        elif blur_method == 'resize':
            blur_method, side_ratio, interpolation = param
            blurred = self.low_res_augmentation(np.array(sample), side_ratio, interpolation)
        else:
            raise ValueError('not a correct blur')

        sample = Image.fromarray(blurred.astype(np.uint8))

        return sample

    def low_res_augmentation(self, img, side_ratio, interpolation):
        # resize the image to a small size and enlarge it back
        img_shape = img.shape
        small_side = int(side_ratio * img_shape[0])
        small_img = cv2.resize(img, (small_side, small_side), interpolation=interpolation)
        interpolation = np.random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        aug_img = cv2.resize(small_img, (img_shape[1], img_shape[0]), interpolation=interpolation)
        return aug_img


