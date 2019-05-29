import numpy as np
import cv2
from itertools import product
from util.logs import get_logger

logger = get_logger('fff')


def segmentation2bbox(image):
    assert len(image.shape) == 2
    flag = np.zeros_like(image, dtype=np.int16)
    s_queue = []
    bboxs = [None]
    delta = list(filter(lambda x: x[0] != 0 or x[1] != 0, product(range(-1, 2), range(-1, 2))))
    for r, c in product(range(image.shape[0]), range(image.shape[1])):
        if flag[r, c]:
            continue
        if image[r, c] == 0:
            continue
        s_queue.append((r, c))
        bbox = (r, r, c, c)
        while len(s_queue) > 0:
            rr, cc = s_queue.pop()
            if flag[rr, cc]:
                continue
            if image[rr, cc] == 0:
                continue
            bbox = (
                min(bbox[0], rr),
                max(bbox[1], rr + 1),
                min(bbox[2], cc),
                max(bbox[3], cc + 1),
            )
            flag[rr, cc] = 1
            for dr, dc in delta:
                if 0 <= rr + dr < image.shape[0] and 0 <= cc + dc < image.shape[1]:
                    s_queue.append((rr + dr, cc + dc))
        bboxs.append(bbox)
    # transfer range to center and width
    bboxs = map(
        lambda x: (
            (x[0] + x[1]) // 2,
            (x[2] + x[3]) // 2,
            (x[1] - x[0]),
            (x[3] - x[2])),
        bboxs[1:])
    bboxs = list(filter(lambda x: x[1] > 0 and x[3] > 0, bboxs))
    return bboxs


if __name__ == '__main__':
    rr = np.zeros((1000, 1000))
    rr[0:4, 1:1000] = 1
    rr[333:334, 333:334] = 1
    rr[250:255, 250:255] = 1
    rr[150:155, 150:155] = 1
    bb = segmentation2bbox(rr > 0)
    print(bb)
