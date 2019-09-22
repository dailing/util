import yaml
from util.logs import get_logger
from abc import ABC, abstractmethod
import pysnooper
from io import StringIO
from collections.abc import Mapping
from abc import ABC, abstractmethod

try:
    import yaml
except ModuleNotFoundError as e:
    yaml = None


logger = get_logger('config')


class NodeMeta(type):
    def __new__(cls, clsname, superclasses, attributedict, **kwargs):
        logger.info(attributedict)
        attributedict['build'] = lambda: make_root_node(
            superclasses[0]._field, attributedict)
        klass = type.__new__(cls, clsname, superclasses, attributedict)
        logger.info(klass)
        return klass


class Node(object, metaclass=NodeMeta):
    def __init__(self, *args, **kwargs):
        logger.info('init node')
        self.args = args
        self.kwargs = kwargs


class Field(ABC):
    def _get_readable_value(self):
        raise Exception('not implemented')

    def _get_bin_value(self):
        raise Exception('not implemented')

    def _set_value(self, value):
        raise Exception('not implemented')

    def _to_dict(self):
        return self._get_readable_value()

    def _from_dict(self):
        raise Exception('not implemented')


class ArgParseField(ABC):
    pass


class ValueField(Field):
    def __init__(self, default=None):
        self.value = default
        self.value_type = type(default)

    def _get_readable_value(self):
        return self.value

    def _get_bin_value(self):
        return self.value

    def _set_value(self, value):
        if not isinstance(None, self.value_type):
            if not isinstance(value, self.value_type):
                logger.warning(
                    f'converting value: {value} of type {type(value)}'
                    f' to type {self.value_type}')
                value = self.value_type(value)
        self.value = value
        return self.value

    def _from_dict(self, value):
        self._set_value(value)


class ConfigField(Field):
    def __init__(
            self, parent=None, name=None, **kwargs):
        # self.parent = parent
        # self.name = name
        logger.info('init')
        self.__dict__['_cfg_storage'] = {}

    # ## Field methods
    def _get_readable_value(self):
        return self

    def _get_bin_value(self):
        return self

    def _set_value(self, value):
        raise Exception("FUCK!")

    def __len__(self):
        return len(self._cfg_storage)

    # ## setattr and getattr
    def __getattr__(self, name):
        logger.info(f'get attr {name}')
        try:
            return self._cfg_storage[name]._get_bin_value()
        except KeyError:
            raise AttributeError()

    def __setattr__(self, name, value):
        try:
            if isinstance(value, Field):
                self._cfg_storage[name] = value
            else:
                self._cfg_storage[name]._set_value(value)
        except KeyError:
            raise AttributeError()

    def _to_dict(self):
        retval = {}
        for k, v in self._cfg_storage.items():
            retval[k] = v._to_dict()
        return retval

    def dump_yaml(self, **kwargs):
        if yaml is None:
            raise Exception(f'Please install pyyaml to use this function')
        return yaml.safe_dump(self._to_dict(), **kwargs)

    def dump_json(self, **kwargs):
        return json.dumps(self._to_dict())


class Config(Node):
    _field = ConfigField


class Value(Node):
    _field = ValueField


class ListValueField(ValueField):
    """
        element_func returns an standard element.
    """
    def __init__(self, element=Value, **kwargs):
        self.fields = []
        self.element_func = element._field

    def append(self, value=None):
        ele = self.element_func()
        logger.info(ele)
        if value is not None:
            ele._from_dict(value)
        self.fields.append(ele)
        return self

    def __add__(self, other):
        self.fields += other
        return self

    def _from_dict(self, dd):
        assert type(dd) is list
        self.fields = []
        for i in dd:
            self.append(i)

    def _to_dict(self):
        return [i._to_dict() for i in self.fields]

    def __getitem__(self, key):
        assert type(key) == int
        item = self.fields[key]
        return item._get_bin_value()

    def _get_readable_value(self):
        return [i._get_readable_value() for i in self.fields]

    def _get_bin_value(self):
        return self

    def __len__(self):
        return self.fields.__len__()

    def get(self):
        return self


class MappingValueField(ValueField):
    def __init__(self, default=None, map_dict=None):
        self.value = default
        self.map_dict = map_dict

    def _get_bin_value(self):
        return self

    def __getattr__(self, key):
        if key not in self.map_dict:
            raise AttributeError(
                f'attribute {key} not exist, '
                f'avaliable keys are '
                f'{list(self.map_dict.keys())}')
        return self.map_dict[key]


class ValueList(Node):
    _field = ListValueField


class ValueMap(Node):
    _field = MappingValueField


def make_node(node_instance, attribute_dict, arg_prefix=''):
    if arg_prefix != '':
        arg_prefix += '.'
    logger.info('node')
    for name, ndef in attribute_dict.items():
        if not isinstance(ndef, Node):
            continue
        logger.info(f'new node {name}')
        new_node = ndef._field(*ndef.args, **ndef.kwargs)
        make_node(new_node, ndef.__class__.__dict__, arg_prefix+name)
        logger.info(f'setting node {name}, {type(new_node)}')
        setattr(node_instance, name, new_node)
    return node_instance


def make_root_node(field_type, attribute_dict):
    node = field_type()
    return make_node(node, attribute_dict, 'cfg')
