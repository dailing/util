import multiprocessing
import os
import pickle
import zlib

from util.logs import get_logger
import random
from collections import Iterable
from typing import Callable
import traceback

try:
	import pymongo
	import gridfs
except ImportError as e:
	pymongo = None

logger = get_logger('Processing')


class TaskFunc():
	def init_env(self):
		pass

	def __call__(self, *arg, **kwargs):
		pass


class ProcessPool(object):
	class DelayedResult:
		def __init__(self, task_id, parent):
			self.task_id = task_id
			self.parent = parent

		def get(self):
			return self.parent._get_result(self.task_id)

	def __init__(self, func, num_thread=None):
		super().__init__()
		self.func = func
		assert callable(self.func)
		self.num_thread = num_thread if num_thread is not None \
			else multiprocessing.cpu_count()
		self.process_pool = None
		self.task_queue = multiprocessing.Queue()
		self.result_queue = multiprocessing.Queue()
		self.slots = multiprocessing.Semaphore(0)
		self.result_map = {}

	@staticmethod
	def _worker_(
			worker_id: int,
			func: TaskFunc,
			task_queue: multiprocessing.Queue,
			result_queue: multiprocessing.Queue,
			semaphore: multiprocessing.Semaphore):
		"""
		:param worker_id: The id of this worker, 1~n
		:param func:
		:param task_queue: pull tasks from this task queue
		:param result_queue:
		:param semaphore: signal this if worker are free
		:return: None
		This pool are designed so that the main process
		will block if all workers are working. So this
		guarantees that not too many tasks are pushed into
		the task_queue, which sometimes can cause problem.
		"""
		logger.debug(f'started worker: {worker_id}')
		try:
			func.init_env()
		except AttributeError:
			pass
		while True:
			# first release the semaphore so that the
			# producer can be unblocked and push data
			# into the task queue
			semaphore.release()
			# retrieve data from task queue if any
			param = task_queue.get()
			# if None is send via task queue, then quit
			if param is None:
				logger.debug(f'stop worker: {worker_id}')
				break
			args, kwargs, task_id = param
			logger.debug(f'received task, id:{task_id}')
			try:
				result = func(*args, **kwargs)
				if task_id is not None:
					# If passed a None task_id, ignore the result
					result_queue.put((task_id, result))
			except Exception as e:
				result_queue.put((task_id, None))
				logger.error(e)
				traceback.print_stack()

	def start(self):
		"""
		start the process
		:return:
		"""
		if self.process_pool is None:
			self.process_pool = []
			for i in range(self.num_thread):
				self.process_pool.append(
					multiprocessing.Process(
						target=ProcessPool._worker_,
						args=(
							i,
							self.func,
							self.task_queue,
							self.result_queue,
							self.slots)))
			for p in self.process_pool:
				p.start()

	def _get_result(self, task_id):
		"""
		retrieve the result of a task
		if the result is not available, stall
		if the result already ready, return it
		:param task_id:
		:return: result
		"""
		try:
			result = self.result_map.pop(task_id)
			if type(result) is Exception:
				return None
			return result
		except KeyError:
			while True:
				tid, result = self.result_queue.get()
				if isinstance(result, Exception):
					logger.error(result)
					result = None
				if tid == task_id:
					return result
				else:
					self.result_map[tid] = result

	def _call(self, task_id, *args, **kwargs):
		self.slots.acquire()
		self.task_queue.put((args, kwargs, task_id))

	def delayed_call(self, *args, **kwargs):
		task_id = str(random.randint(0, 1000000000))
		self._call(task_id, *args, **kwargs, )
		return ProcessPool.DelayedResult(task_id, self)

	def call_without_result(self, *args, **kwargs):
		self._call(None, *args, **kwargs)

	def stop(self):
		logger.info('stopping task queue!')
		for p in self.process_pool:
			self.slots.acquire()
			self.task_queue.put(None)
		for p in self.process_pool:
			p.join()


def parallel_map(
		func: Callable,
		iter_obj: Iterable,
		pool_size=50,
		num_thread=None) -> Iterable:
	"""
	maps a iterable to another parallel.
	:param func: map function
	:param iter_obj: input
	:param pool_size: size of the result pool, since some item are run
					  parallel, some item can finish earlier then
					  others, with a pool, the early result can be put
					  in the pool and run the next. so that we can max
					  cpu usage without waiting the slower tasks.
	:return: output of the function on each item in the input
	"""
	tasks = ProcessPool(func, num_thread=num_thread)
	tasks.start()
	pool = []
	x = iter_obj.__iter__()
	# insert as many as possiable if the pool is not full
	# TODO optimize the logic here
	while True:
		try:
			item = x.__next__()
			pool.append(tasks.delayed_call(item))
			if len(pool) > pool_size:
				result = pool.pop(0)
				yield result.get()
		except StopIteration:
			# logger.info(f'left pool {len(pool)}')
			try:
				result = pool.pop(0)
				yield result.get()
				# logger.info("HERE")
			except IndexError:
				# logger.info('calling stop')
				tasks.stop()
				return
	tasks.stop()


def list_to_map_of_list(key_func: Callable, l: Iterable) -> dict:
	result = {}
	for i in l:
		key = key_func(i)
		if key in result:
			result[key].append(i)
		else:
			result[key] = [i]
	return result


def run_once(f):
	"""
	this is a decorator for functions.
	the returned function will only run once.
	:param f: the function
	:return: the wrapper function that will only run once
	"""

	def wrapper(*args, **kwargs):
		if not wrapper.has_run:
			wrapper.has_run = True
			return f(*args, **kwargs)

	wrapper.has_run = False
	return wrapper


def run_if(eval_func):
	def replace_func(f):
		def wrapper(*args, **kwargs):
			if eval_func():
				return f(*args, **kwargs)

		return wrapper

	return replace_func


def parallel_process(
		func: Callable,
		iter_obj: Iterable) -> None:
	"""
	process each input with func and ignore result
	:param func: function to perform the process
	:param iter_obj: input object
	:return:
	"""
	task = ProcessPool(func)
	task.start()
	for i in iter_obj:
		task.call_without_result(i)
	task.stop()


def bin_filter(func: Callable, iter: Iterable):
	poss = []
	neg = []
	for i in iter:
		if func(i):
			poss.append(i)
		else:
			neg.append(i)
	return poss, neg


def head(n: int, iter: Iterable):
	iter = iter.__iter__()
	while n > 0:
		n -= 1
		yield iter.__next__()


if __name__ == '__main__':
	def test_run_once():
		class testRunOnce:
			def __init__(self):
				self.accumulator = 0

			@run_once
			def runOnce(self):
				self.accumulator += 1

		ttt = testRunOnce()
		for i in range(10):
			ttt.runOnce()
		assert ttt.accumulator == 1


	test_run_once()


class Identity():
	"""Identity function, return as it is"""

	def __init__(self):
		super().__init__()

	def transform(self, img):
		return img


class MongoConnector(TaskFunc):
	def __init__(self, host='10.10.0.1', port=27017, username='root', password='example'):
		self.db = None
		self.gfs = None
		self.host = host
		self.port = port
		self.username = username
		self.password = password

	def init_env(self):
		logger.info(f'connecting to {self.host}:{self.port}')
		self.db = pymongo.MongoClient(
			self.host,
			self.port,
			username=self.username,
			password=self.password).fundus_images
		self.gfs = gridfs.GridFS(self.db)


class MongoFileReader(MongoConnector, TaskFunc):
	def __init__(self, *args, **kwargs):
		super(MongoFileReader, self).__init__(*args, **kwargs)
		self.init_env()

	def __call__(self, image_name):
		try:
			logger.debug(f'fetching {image_name}')
			f = self.gfs.find_one({'filename': image_name})
			if f is None:
				# logger.warning(f'Missing:{image_name}')
				return None
			return f.read()
		except Exception as e:
			logger.error(e)
			return None


class MongoFileWriter(MongoConnector, TaskFunc):
	def __init__(self, *args, **kwargs):
		super(MongoConnector, self).__init__(*args, **kwargs)

	def __call__(self, file_content, name, source='None'):
		self.init_env()
		if type(file_content) is not bytes:
			logger.error('Input Type not correct!'
						 f' Except byters, got {type(file_content)}')
			traceback.print_stack()
		try:
			logger.debug(f'saving {name}')
			if self.gfs.exists(filename=name):
				logger.error(f'file \'{name}\' already exists.')
				return None
			self.gfs.put(
				file_content,
				filename=name,
				source=source,
			)
		except Exception as e:
			logger.error(e)
			traceback.print_stack()
			return None
		return name


class NormalFileReader(TaskFunc):
	def init_env(self):
		super().__init__()

	def __call__(self, args):
		try:
			file_content = open(args, 'rb').read()
			return file_content
		except Exception as e:
			logger.error(e)
			return None


def mkdirParentDir(filename):
	dirname = os.path.dirname(filename)
	if not os.path.exists(dirname):
		logger.info(f'making new dir:{dirname}')
		try:
			os.makedirs(dirname)
		except Exception as e:
			logger.critical(f'Error making the fucking directory:{dirname}!')
			return False
	return True


def compressedDump(obj, filename, level=9):
	mkdirParentDir(filename)
	data = pickle.dumps(obj)
	data = zlib.compress(data, level)
	with open(filename, 'wb') as f:
		f.write(data)


def compressedLoad(filename):
	with open(filename, 'rb') as f:
		data = f.read()
		data = zlib.decompress(data)
		obj = pickle.loads(data)
		return obj


def MapReduce(reduceFunc, mapFunc, data,
		mapTransfer=None,
		reduceTransfer=None,
		parallelMap = False
	):
	'''
	Args:
		reduceFunc (callable): reduce finction.
			must recieve 2 parameters, and the order
			of these parameters does not matter
		mapFunc (callable): map function, map the data
			to a key, value pair, values in each key 
			will be collected by reduce.
		data (any): Input Data to process
		mapTransfer(callable): called on original data,
			if not none, each data will be transfered
			using the mapTransfer.
		reduceTransfer(callable): if not None, the result
			after map stage will be transfered using this
			reduceTransfer.
	Returns:
		dict: result
	'''
	if parallelMap:
		mmap = parallel_map
	else:
		mmap = map
	if mapTransfer is not None:
		data = map(mapTransfer, data)
	mapResult = mmap(mapFunc, data)
	result = {}
	for key, value in mapResult:
		if reduceTransfer is not None:
			value = reduceTransfer(value)
		if key in result:
			result[key] = reduceFunc(result[key], value)
		else:
			result[key] = value
	return result


class CachedProperty(object):
	"""
	Example of Usage:
		class Foo(object):
		def __init__(self):
			print('fuck init')


		@CachedProperty
		def bar(self):
			print('asdfhas')
			return "fjhgfjgfj"

		foo = Foo()
		print("initiate ... ")
		# print(foo.bar)
		# print(foo.bar)	
	"""
	def __init__(self, func, name=None):
		self.func = func
		self.name = name if name is not None else func.__name__
		self.__doc__ = func.__doc__

	def __get__(self, instance, class_):
		if instance is None:
			return self
		res = self.func(instance)
		setattr(instance, self.name, res)
		return res


