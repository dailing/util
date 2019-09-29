import yaml
from util.logs import get_logger
from abc import ABC, abstractmethod
import pysnooper
from io import StringIO

logger = get_logger('fuck')


class Field(object):
    def __init__(**kwargs):
        pass


class ValueField(Field):
    def __init__(self, default_value=None, required=False, **kwargs):
        self.default_value = default_value
        self.required = required

    def get(self):
        return self.default_value

    def set(self, value):
        self.default_value = value

    def from_dict(self, value):
        self.set(value)

    def to_dict(self):
        return self.get()


class NameMappingField(ValueField):
    def __init__(self, default_value, mapping=None, **kwargs):
        super().__init__(default_value, **kwargs)
        self.mapping = mapping

    def get(self):
        return self.mapping[self.default_value]

    def set(self, value):
        self.default_value = value

    def to_dict(self):
        return self.default_value

    def from_dict(self, value):
        self.set(value)


class ListOfField(Field):
    """
        element_func returns an standard element.
    """
    def __init__(self, element_func: callable = None, **kwargs):
        self.fields = []
        self.element_func = element_func

    # @pysnooper.snoop()
    def append(self, value=None):
        ele = self.element_func()
        logger.info(ele)
        if value is not None:
            ele.from_dict(value)
        self.fields.append(ele)
        return self

    def __add__(self, other):
        self.fields += other
        return self

    def from_dict(self, dd):
        assert type(dd) is list
        self.fields = []
        for i in dd:
            self.append(i)

    def to_dict(self):
        result = []
        for i in self.fields:
            result.append(i.to_dict())
        return result

    # @pysnooper.snoop()
    def __getitem__(self, key):
        assert type(key) == int
        item = self.fields[key]
        # print(item)
        # print(hasattr(item, 'get'))
        if hasattr(item, 'get'):
            return item.get()
        return item

    def __len__(self):
        return self.fields.__len__()

    def get(self):
        return self


class Configure():
    def __init__(self, parent=None):
        self.__dict__['_cfg_storage'] = {}
        self.__dict__['parent'] = parent
        self.__dict__['root'] = self
        if self.parent is None:
            self.root = self
        else:
            self.root = self.parent.root

    def add(self, name, default_value=None, **kwargs):
        if name not in self._cfg_storage:
            self._cfg_storage[name] = ValueField(
                default_value=default_value, **kwargs)
            return self
        else:
            raise Exception('name already exits!')

    def add_multi(self, **values):
        for k, v in values.items():
            if type(v) is not dict:
                v = dict(default_value=v)
            self.add(k, **v)
        return self

    def add_list(self, name, ele_func, **kwargs):
        self._cfg_storage[name] = ListOfField(element_func=ele_func, **kwargs)
        return self

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
        else:
            raise AttributeError()

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
            if not hasattr(self._cfg_storage[k], 'from_dict'):
                self.__setattr__(k, v)
            else:
                self._cfg_storage[k].from_dict(v)

    def from_yaml(self, file):
        if type(file) is str:
            file = open(file, 'r')
        dd = yaml.safe_load(file)
        self.from_dict(dd)

    def to_yaml(self, file_to_write=None):
        if type(file_to_write) is str:
            file_to_write = open(file_to_write, 'w')
        return yaml.safe_dump(self.to_dict(), file_to_write)

    def make_sample_yaml(self, file=None):
        for k, v in self._cfg_storage.items():
            if isinstance(v, Configure):
                v.make_sample_yaml()
            elif isinstance(v, ListOfField) and len(v) == 0:
                v.append({})
        return self.to_yaml(file)

    def _parent(self):
        if self._parent is None:
            raise Exception('Root node has no parent')
        return self._parent
