from struct import unpack, iter_unpack
from io import BytesIO
from collections import namedtuple
import numpy as np
# from . import logs
from os.path import join as pjoin
from functools import reduce
from torch.utils.data import Dataset

# logger = logs.get_logger('mnist dataset')

_data_type_ = namedtuple('_data_type_', ['npType', 'pyFormat', 'nbytes'])
_data_type_map_ = {
    0x08: _data_type_(np.uint8, '>B', 1),  # unsigned byte
    0x09: _data_type_(np.int8, '>b', 1),  # signed byte
    0x0B: _data_type_(np.uint16, '>h', 2),  # short (2 bytes)
    0x0C: _data_type_(np.int16, '>i', 4),  # int (4 bytes)
    0x0D: _data_type_(np.float32, '>f', 4),  # float (4 bytes)
    0x0E: _data_type_(np.float64, '>d', 8),  # double (8 bytes)
}


def parseIDXFile(payload):
        if type(payload) is bytes:
            payload = BytesIO(payload)
        else:
            payload = payload
        rawByteLength = payload.seek(0, 2)
        payload.seek(0, 0)

        assert payload is not None
        if payload.tell() >= rawByteLength:
            return None
        magicNumber = payload.read(2)
        assert magicNumber[0] == 0, f'magical number error, read: {magicNumber[0]:X}'
        assert magicNumber[1] == 0, f'magical number error, read: {magicNumber[1]:X}'
        typeCode = payload.read(1)[0]
        assert typeCode in _data_type_map_, \
            f'data type 0x{typeCode:02X} not recognized'
        dtype = _data_type_map_[typeCode]
        ndim = int(payload.read(1)[0])
        sizes = tuple(i[0] for i in iter_unpack(f'>i', payload.read(ndim * 4)))
        payloadSize = dtype.nbytes * reduce(lambda x, y: x * y, sizes)
        # arr = unpack(dtype.pyFormat, )
        dt = np.dtype(dtype.pyFormat)
        npArr = np.frombuffer(
            payload.read(payloadSize),
            dtype=dt,
        )
        npArr = npArr.reshape(sizes)
        return npArr


class Mnist(Dataset):

    fileNameMap = {
        'train_data': 'train-images-idx3-ubyte',
        'train_label': 'train-labels-idx1-ubyte',
        'test_label': 't10k-labels-idx1-ubyte',
        'test_data': 't10k-images-idx3-ubyte',
    }

    def __init__(self, root='/data/mnist', split='train'):
        imageFilename = pjoin(root, Mnist.fileNameMap[f'{split}_data'])
        labelFilename = pjoin(root, Mnist.fileNameMap[f'{split}_label'])
        self.images = parseIDXFile(open(imageFilename, 'rb'))
        self.labels = parseIDXFile(open(labelFilename, 'rb'))

    def __getitem__(self, index):
        return self.images[index, :, :], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


if __name__ == '__main__':
    mnist = Mnist()
    print(len(mnist))
    print(mnist[0])
