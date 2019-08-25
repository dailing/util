import yaml
from util.logs import get_logger
from abc import ABC, abstractmethod
import pysnooper
from io import StringIO

logger = get_logger('fuck')


class Field(object):
    def __init__(self, default_value=None, required=False, **kwargs):
        self.default_value = default_value
        self.required = required

    def get(self):
        return self.default_value

    def set(self, value):
        self.default_value = value

    def to_dict(self):
        return self.get()


class ValueField(Field):
    pass


class NameMappingField(Field):
    def __init__(self, default_value, mapping=None, **kwargs):
        super().__init__(default_value, **kwargs)
        self.mapping = mapping

    def get(self):
        return self.mapping[self.default_value]

    def set(self, value):
        self.default_value = value

    def to_dict(self):
        return self.default_value


class Configure():
    def __init__(self, parent=None):
        self.__dict__['_cfg_storage'] = {}
        self.__dict__['parent'] = parent
        self.__dict__['root'] = self
        if self.parent is None:
            self.root = self
        else:
            self.root = self.parent.root

    def add(self, name, default_value=None):
        if name not in self._cfg_storage:
            self._cfg_storage[name] = ValueField(default_value)
            return self
        else:
            raise Exception('name already exits!')

    def add_mapping(self, name, name_mapping, default_value=None, **kwargs):
        if name in self._cfg_storage:
            raise Exception('name already axists')
        self._cfg_storage[name] = NameMappingField(
            default_value=default_value,
            mapping=name_mapping,
            **kwargs
        )
        return self

    # @pysnooper.snoop()
    def add_alias(self, name, original_name):
        if name not in self._cfg_storage:
            new_val = self.get_absolute(original_name)
            self._cfg_storage[name] = new_val
        else:
            logger.error('name already exists!')

    def add_subconfigure(self, name):
        self._cfg_storage[name] = Configure(parent=self)
        return self._cfg_storage[name]

    # @pysnooper.snoop()
    def __getattr__(self, name):
        if name in self._cfg_storage:
            val = self._cfg_storage[name]
            if issubclass(val.__class__, Field):
                return val.get()
            elif issubclass(val.__class__, Configure):
                return val
            else:
                logger.error(f'got {type(val)}')

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
            return
        elif name in self._cfg_storage:
            val = self._cfg_storage[name]
            if issubclass(val.__class__, Field):
                val.set(value)
            else:
                logger.error(f'cannot set value \
                    of item {name}, type {type(name)}')

    def get_absolute(self, name):
        root = self.root
        for i in name.split('.'):
            root = root._cfg_storage[i]
        return root

    def to_dict(self):
        result = {}
        for k, v in self._cfg_storage.items():
            result[k] = v.to_dict()
        return result

    def from_dict(self, cfg):
        for k, v in cfg.items():
            if k not in self._cfg_storage:
                raise Exception(f'configure {k} not found')
            if isinstance(self._cfg_storage[k], Field):
                self.__setattr__(k, v)
            elif isinstance(self._cfg_storage[k], Configure):
                self._cfg_storage[k].from_dict(v)
            else:
                raise Exception("FUCK YOU!!!")

    def from_yaml(self, file):
        if type(file) is str:
            file = open(file, 'r')
        dd = yaml.safe_load(file)
        self.from_dict(dd)

    def to_yaml(self, file_to_write=None):
        if type(file_to_write) is str:
            file_to_write = open(file_to_write, 'w')
        return yaml.safe_dump(self.to_dict(), file_to_write)
