import random
import numbers
import collections
import cv2

import torch

class Compose(object):
    # Composes segtransforms: segtransform.Compose([segtransform.RandScale([0.5, 2.0]), segtransform.ToTensor()])
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image, label):
        for t in self.segtransform:
            image, label = t(image, label)
        return image, label


class RandRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=0, p=0.5):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
        return image, label


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        return image, label


class RandomGaussianBlur(object):
    def __init__(self, radius=5, p=0.5):
        self.radius = radius
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label


class RGB2BGR(object):
    # Converts image from RGB order to BGR order, for model initialized from Caffe
    def __call__(self, image, label, label2):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label, label2


class BGR2RGB(object):
    # Converts image from BGR order to RGB order, for model initialized from Pytorch
    def __call__(self, image, label, label2):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label, label2
