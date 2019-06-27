import numpy as np
from itertools import product
import math
from util.logs import get_logger


logger = get_logger('safas')


class ImageSlider:

    @staticmethod
    def calculate_start_p(stride, size):
        num_step = math.ceil(size / stride)
        stride = size / num_step
        for i in range(num_step + 1):
            yield math.floor(stride * i)

    def __init__(self, image_shape=None, width=512, overlapping=0.2):
        self.image_shape = image_shape[:2]
        self.width = width
        self.overlapping = overlapping
        self.top_left_points = []
        self.stride = (1 - self.overlapping) * self.width

        def calculate_start_p(stride, size):
            num_step = math.ceil(size / stride)
            stride = size / num_step
            for i in range(num_step + 1):
                yield math.floor(stride * i)

        for row, col in product(
                calculate_start_p(self.stride, image_shape[0] - self.width),
                calculate_start_p(self.stride, image_shape[1] - self.width)):
            row = min(image_shape[0] - self.width, row)
            col = min(image_shape[1] - self.width, col)
            # image_patch.append(img[row:row + size, col:col + size, ::])
            self.top_left_points.append((row, col))

    def split(self, image):
        for row, col in self.top_left_points:
            pitch = image[(
                slice(row, row + self.width),
                slice(col, col + self.width),
                *[slice(0, i) for i in image.shape[2:]])]
            yield pitch

    def combine(self, image_patches):
        restored_image = np.zeros((
            *self.image_shape,
            *image_patches[0].shape[2:]))
        counter = np.zeros(restored_image.shape)
        for patch, (row, col) in zip(image_patches, self.top_left_points):
            logger.debug(restored_image.shape)
            indices = (
                slice(row, row + self.width),
                slice(col, col + self.width),
                *[slice(0, i) for i in self.image_shape[2:]])
            restored_image[indices] += patch
            counter[indices] += 1
        return restored_image / counter
