import time

import numpy as np


class Avger(object):
    """docstring for Avger."""

    def __init__(self):
        super(Avger, self).__init__()
        self.counter = 0
        self.accumulater = None

    def update(self, num):
        if self.counter == 0:
            self.accumulater = num
        else:
            self.accumulater = self.accumulater + num
        self.counter += 1

    def get_avg(self):
        if self.accumulater is not None:
            return self.accumulater / self.counter
        return 0


class WindowedAvger(object):
    """docstring for Avger."""

    def __init__(self, window_size=10):
        super(WindowedAvger, self).__init__()
        self.window_size = window_size
        self.accumulater = []

    def update(self, num):
        self.accumulater.append(num)
        if len(self.accumulater) > self.window_size:
            self.accumulater.pop(0)

    def get_avg(self):
        return np.mean(self.accumulater, axis=0)


class Timer(object):

    def __init__(self):
        self.avger = Avger()
        self.last = None
        self.startTime = 0

    def start(self):
        self.startTime = time.time()

    def step(self):
        now = time.time()
        self.last = now - self.startTime
        self.avger.update(self.last)
        self.startTime = now

    def get_avg(self):
        return self.avger.get_avg()

    def get_last(self):
        return self.last
