# defins the task object 1
import logging

import redis
import json
import string
import random
import time
from abc import ABC, abstractmethod
import traceback
from itertools import chain
import base64
from os.path import exists
import pickle
import sys
import lzma


# adding some optional dependency
try:
    import cv2
except Exception as e:
    cv2 = e

try:
    import numpy as np
except Exception as e:
    np = e

def get_logger(name: str, print_level=logging.DEBUG):
    formatter = logging.Formatter(
        fmt="%(levelname)6s %(asctime)s [%(pathname)s:%(lineno)-3d] %(message)s",
        # datefmt='%H:%M:%S,uuu',
    )
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger('tasks')

random.seed(int(time.time() * 1000000))


def randstr(n):
    return ''.join(
        random.choice(
            string.ascii_letters + string.digits) for _ in range(n))


# server side
class Request(object):
    """docstring for Request."""

    def __init__(self, payload):
        super(Request, self).__init__()
        self.payload = payload

    def get_json(self):
        return json.loads(self.get_string())

    def get_string(self):
        return self.payload.decode('utf-8')

    def get(self):
        return self.payload


class ResponseWriter(object):
    """docstring for ResultWriter."""

    def __init__(self, slot, redisConn):
        super(ResponseWriter, self).__init__()
        self.slot = slot
        self.redisConn = redisConn

    def write_raw(self, payload):
        self.redisConn.lpush(self.slot, payload)

    def write_json(self, obj):
        payload = json.dumps(obj)
        self.write_raw(payload)


class Response(object):
    """docstring for Response."""

    def __init__(self, slot, redisConn):
        super(Response, self).__init__()
        self.slot = slot
        self.redisConn = redisConn
        self.raw = None

    def ready(self):
        r = self.redisConn.llen(self.slot)
        return r == 1

    def get_raw(self):
        if self.raw is None:
            name, payload = self.redisConn.brpop(self.slot)
            name = name.decode('utf-8')
            assert name == self.slot
            logger.debug('getting key from + ' + name)
            self.raw = payload
            self.raw = lzma.decompress(self.raw)
        return self.raw

    def get_json(self):
        obj = None
        obj = json.loads(self.get_raw().decode('utf-8'))
        return obj

    def get(self):
        return pickle.loads(self.get_raw())


class Task(object):
    """docstring for Task."""

    def __init__(self, task_name=None, redis_host='localhost', redis_port=6379):
        super(Task, self).__init__()
        self.payload_channel = task_name + '.input_channel'
        self.output_slot = task_name + '.output_slot'
        self.wait_slot = task_name + '.wait_slot'
        self.task_name = task_name
        counter=0
        while True:
            try:
                pool = redis.ConnectionPool(host=redis_host, port=redis_port, db=0)
                self.redisConn = redis.Redis(connection_pool=pool)
            except Exception as e:
                logger.info(f'error connection, retrying {counter} times')
                time.sleep(5)
                continue
            break

    def issue(self, input):
        '''
            Issue a task
        '''
        input = pickle.dumps(input)
        input = lzma.compress(input)
        pipe = self.redisConn.pipeline(transaction=True)
        result_slot = self.task_name + '.output_slot.' + randstr(20)
        pipe.lpush(self.payload_channel, input)
        pipe.lpush(self.output_slot, result_slot)
        pipe.lpush(self.wait_slot, 0)
        result = pipe.execute()
        assert result[0] == result[1]
        # TODO check output value
        # TODO return a value
        logger.debug('setting return slot:' + result_slot)
        return Response(result_slot, self.redisConn)

    def wait_for_task(self):
        '''
            wait for a task, return raw data from queue
        '''
        self.redisConn.brpop(self.wait_slot)
        pipe = self.redisConn.pipeline(transaction=True)
        pipe.brpop(self.payload_channel)
        pipe.brpop(self.output_slot)
        output = pipe.execute()
        slot = ''
        payload = ''
        for k, v in output:
            k = k.decode('utf-8')
            if k == self.payload_channel:
                payload = v
            elif k == self.output_slot:
                slot = v
            else:
                raise TypeError()
        return Request(payload), ResponseWriter(slot, self.redisConn)


# any service should extend this class
# and then it can be passed to the fucking
# MService as init parameter
class MServiceInstance(ABC):
    @abstractmethod
    def init_env(self):
        pass

    @abstractmethod
    def __call__(self, arg):
        pass


class MService:
    def __init__(self, serviceName=None, service=None,
                 inputTransform=None, outputTransform=None):
        """
        constructor of MService
        :param serviceName: name of the service
        :param service: service Instance.
        :param inputTransform: dict, string key,
            value should be any callable object
        :param outputTransform: same as inputTransform
        """
        if serviceName is None:
            serviceName = randstr(20)
        if inputTransform is None:
            inputTransform = {}
        if outputTransform is None:
            outputTransform = {}
        self.serviceName = serviceName
        self.service = service
        self.inputTransform = inputTransform
        self.outputTransform = outputTransform
        # check if the transform function is callable
        for k, v in chain(inputTransform.items(), outputTransform.items()):
            if not hasattr(v, '__call__'):
                logger.error(f"transform object of service{serviceName},"
                             f"key{k} is not callable")
                raise Exception("FUCK ME!")

    def run(self):
        # check service
        if self.service is None:
            logger.error("service object is None")
            return
        if not issubclass(self.service.__class__, MServiceInstance):
            logger.error("Service object is not subclass of MService")
            return
        logger.info(f"init service: {self.serviceName}")
        try:
            self.service.init_env()
        except Exception as e:
            logger.error(e)
            traceback.print_tb(e.__traceback__)
        task = Task(self.serviceName, 'redis', 6379)
        logger.info('task ready....')
        while True:
            result = None
            resp_writer = None
            try:
                request, resp_writer = task.wait_for_task()
                logger.info('received task')
                reqJson = pickle.loads(request.get())
                result = self.service(reqJson)
                # logger.debug(f'result is {result}')
            except Exception as e:
                logger.error(e, exc_info=True)
                continue
            finally:
                # write to the task slot anyway.
                if resp_writer is not None:
                    try:
                        resp_writer.write_raw(pickle.dumps(result))
                    except Exception as e:
                        logger.error(e, exec_info=True)


class bin2b64:
    def __init__(self):
        pass

    def __call__(self, binData):
        if type(binData) is str:
            binData = binData.encode('utf-8')
        output = base64.encodebytes(binData)
        return output.decode('utf-8')


class b642Image:
    def __init__(self):
        pass

    def __call__(self, b64str):
        try:
            encoded_img = base64.b64decode(b64str)
            encoded_img = np.frombuffer(encoded_img, np.uint8)
            img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(e)
            return None
        else:
            return img

class b642Bytes():
    """docstring for b642Bytes"""
    def __init__(self):
        pass

    def __call__(self, b64str):
        try:
            encoded_img = base64.b64decode(b64str)
            return encoded_img
        except Exception as e:
            return None

class Image2Bytes(object):
    """docstring for Image2Bytes"""
    def __init__(self, encodeType='.png'):
        super(Image2Bytes, self).__init__()
        self.encodeType = encodeType
    
    def __call__(self, img):
        succ, code = cv2.imencode(self.encodeType, img)
        if not succ:
            logger.error('encode not successful')
            return None
        return code.tobytes()

class listMap:
    def __init__(self, transformObj = None):
        self.transform = transformObj
        if not hasattr(self.transform, '__call__'):
            logger.error("transform object is not callable")
            raise TypeError("transform object is not callable")

    def __call__(self, listOfObj):
        return [self.transform(i) for i in listOfObj]


class CallWithImages:
    def __init__(self, taskName):
        logger.info(f"task name {taskName}")
        self.task = Task(taskName, 'redis')
        self.trans = bin2b64()

    def predict(self, images):
        if type(images) is not list:
            images = [images]
        new_list = []
        for i in images:
            new_list.append(self.trans(i))
        response = self.task.issue_json(dict(data = new_list))
        resp = response.get_raw()
        return resp.decode('utf-8')


def predict_file_list(taskName, file_list, resultfile=None):
    from tf.tfutil import MongoFileReader
    from tqdm import tqdm
    if resultfile is None:
        counter = 0
        while True:
            counter += 1
            resultfile = f'result_for_task_{taskName}.{counter}.txt'
            if not exists(resultfile):
                break
    filereader = MongoFileReader()
    filereader.init_env()
    service = CallWithImages(taskName)
    if not exists(file_list):
        logger.error(f"file: {file_list} does not exists")
        return
    with open(file_list, 'r') as f:
        names = f.read().splitlines()
    binfiles = map(filereader, names)

    with open(resultfile, 'w') as ofile:
        for originalFileName, binfile in zip(names, binfiles):
            if binfile is None:
                continue
            resp = service.predict(binfile)
            logger.info(f'{originalFileName}\t{resp}')
            ofile.write(f'{originalFileName}\t{resp}\n')


if __name__ == '__main__':
    import fire
    fire.Fire(predict_file_list)
