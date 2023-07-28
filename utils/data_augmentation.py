import random
import cv2
import numpy as np


class RandomHorizontalFlip(object):
    """随机左右翻转"""

    def __call__(self, image_x):
        p = random.random()
        if p < 0.5:
            new_image_x = cv2.flip(image_x, 1)
            return new_image_x
        else:
            return image_x


class RandomRotation(object):
    """随机旋转，最后旋转"""

    def __init__(self, angle=10):
        self.angle = angle

    def __call__(self, image_x):
        angle = random.randint(-self.angle, self.angle)
        center = (image_x.shape[1] // 2, image_x.shape[0] // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
        new_image_x = cv2.warpAffine(image_x, rot_mat, (image_x.shape[1], image_x.shape[0]))
        return new_image_x


class Resize(object):
    def __init__(self, size):
        self.size = size

    """调整图片尺寸"""
    def __call__(self, image_x):
        return cv2.resize(image_x, [self.size, self.size])


class RandomBlur(object):
    """随机模糊"""

    def __init__(self, blur_size=32):
        self.blur_size = blur_size

    def __call__(self, image_x):
        p = random.random()
        if p < 0.5:
            h, w = image_x.shape[0], image_x.shape[1]
            return cv2.resize(cv2.resize(image_x, (self.blur_size, self.blur_size)), (h, w))
        return image_x


class RandomScale(object):
    """先随机缩放，再随机移动"""
    def __init__(self, offset=8):
        self.offset = offset

    def __call__(self, image_x):
        offset = random.randint(0, self.offset)
        if offset == 0:
            return image_x
        h, w = image_x.shape[1], image_x.shape[0]
        return cv2.resize(image_x[offset:-offset, offset:-offset, :], (h, w))


class RandomMove(object):
    def __init__(self, offset=8):
        self.offset = offset

    def __call__(self, image_x):
        tranMat = np.float32(
            [[1, 0, random.randint(-self.offset, self.offset)], [0, 1, random.randint(-self.offset, self.offset)]])
        dimensions = (image_x.shape[1], image_x.shape[0])
        return cv2.warpAffine(image_x, tranMat, dimensions)


class Normalization(object):
    """
        same as mxnet, normalize into [-1, 1]
    """

    def __call__(self, img):
        return ((img / 255) - 0.5) / 0.5
