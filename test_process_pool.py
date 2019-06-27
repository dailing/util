import unittest
import numpy as np

from util.process_pool import CachedFunction
from util.image_process import ImageSlider
from util.logs import get_logger
import matplotlib.pyplot as plt
import hashlib

logger = get_logger('fuck')


class TestCach edFunc(unittest.TestCase):
    def test_int_single_key(self):
        @CachedFunction(cache_dir='/tmp/test/cache')
        def sqrt(x):
            return x ** 2
        for i in range(100):
            self.assertEqual(sqrt(i), i ** 2)
        for i in range(100):
            self.assertEqual(sqrt(i), i ** 2)
        self.assertEqual('fuck', 'fuck')

    def test_cached_func_file_input(self):
        np.random.seed(5153424)
        bytes_arr = [np.random.bytes(1024*325) for i in range(10000)]
        md5_arr = []
        for i in bytes_arr:
            md5 = hashlib.md5()
            md5.update(i)
            md5_arr.append(md5.hexdigest())

        @CachedFunction(cache_dir='/tmp/test/cache')
        def calculate_md5(arr=None):
            md5 = hashlib.md5()
            md5.update(arr)
            return md5.hexdigest()
        for arg, result in zip(bytes_arr, md5_arr):
            calcu = calculate_md5(arg)
            self.assertEqual(calcu, result)


class TestImageSplit(unittest.TestCase):
    def test_split_and_recover(self):
        rand_img = np.random.rand(2313, 3242, 3)
        slider = ImageSlider(rand_img.shape)
        images = list(slider.split(rand_img))
        self.assertTrue(np.all(rand_img[:512, :512, :] == images[0]))
        restored_image = slider.combine(images)
        self.assertTrue(np.all(restored_image == rand_img))


if __name__ == "__main__":
    unittest.main()
