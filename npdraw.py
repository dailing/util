import numpy as np
import math
import cairo


def _fromnumpy(img):
    if np.issubdtype(img.dtype, np.floating):
        img = (img * 255).astype(np.uint8)
    assert img.dtype == np.uint8, f'{img.dtype}'
    img = img[:, :, ::-1]
    img = np.dstack((img, np.zeros(img.shape[:2], dtype=np.uint8)))
    img = img.copy()
    print(img.shape)
    print(img.strides)
    surface = cairo.ImageSurface.create_for_data(
        img, cairo.FORMAT_ARGB32, img.shape[1], img.shape[0])
    return surface


def _tonumpy(surface: cairo.ImageSurface):
    np_image = np.ndarray(
        shape=(surface.get_height(), surface.get_width(), 4),
        dtype=np.uint8,
        buffer=surface.get_data(),
        strides=(surface.get_width() * 4, 4, 1)
    )
    np_image = np_image[:, :, :-1]
    np_image = np_image[:, :, ::-1]
    np_image = np_image.copy()
    return np_image


def draw_bounding_box(img: np.ndarray, crow, ccol, rowrange, colrange):
    H, W, C = img.shape
    surface = _fromnumpy(img)
    ctx = cairo.Context(surface)
    ctx.scale(W, H)

    ctx.rectangle(
        (ccol - colrange / 2) / W,
        (crow - rowrange / 2) / H,
        colrange / W,
        rowrange / H,
    )  # Rectangle(x0, y0, x1, y1)
    ctx.set_source_rgba(0, 1, 0, 1)
    ctx.set_line_width(1/H)
    ctx.set_dash([5/H])
    ctx.stroke_preserve()
    ctx.set_source_rgba(1,0,0,0.2)
    ctx.fill()
    return _tonumpy(surface)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from voc2012 import VOC
    import random

    voc = VOC(split='train')
    for i in range(20):
        # n = 310
        n = random.randint(0, len(voc))
        img, xx = voc[n]
        for x in xx:
            img = draw_bounding_box(img, *x)
        plt.figure()
        plt.imshow(img)
        plt.show()
    # WIDTH, HEIGHT = 256, 256
