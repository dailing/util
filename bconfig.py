import yaml
from util.logs import get_logger
from abc import ABC, abstractmethod
from io import StringIO
from collections.abc import Mapping
from abc import ABC, abstractmethod
import argparse
import json

try:
    import yaml
except ModuleNotFoundError as e:
    yaml = None


logger = get_logger('config')


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
    def __init__(self):
        self.__dict__['__arg_parse_param_name'] = None

    def _set_arg_parse_param_name(self, name):
        self.__arg_parse_param_name = name

    def _get_arg_parse_param_name(self):
        return self.__arg_parse_param_name

    def _get_arg_parse_kwargs(self):
        return {}


class ValueField(Field, ArgParseField):
    def __init__(self, default=None):
        super().__init__()
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


class ArgParseAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        logger.info(f'{values}, {self.dest}')
        if not hasattr(namespace, '_dict_storage'):
            namespace._dict_storage = {}
        namespace._dict_storage[self.dest] = values


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

    def _walk(self):
        yield self
        for k, v in self._cfg_storage.items():
            if isinstance(v, ConfigField):
                for xx in v._walk():
                    yield xx
            else:
                yield v

    def parse_args(self, args=None, namespace=None):
        """
        parse command line args, and update args in this config
        """
        parser = argparse.ArgumentParser()
        args_map = {}
        for field in self._walk():
            if isinstance(field, ArgParseField):
                name = field._get_arg_parse_param_name()
                logger.info(name)
                args_map[name] = field
        args_alies = {k: [k] for k in args_map}
        taken_args = set(args_alies)
        for k in args_map:
            sep_fields = k.split('.')
            short_cut = sep_fields.pop()
            while(short_cut in taken_args and len(sep_fields) > 0):
                short_cut = f'{sep_fields.pop()}.{short_cut}'
            if len(sep_fields) > 0:
                taken_args.add(short_cut)
                args_alies[k].append(short_cut)
                if short_cut[0] not in taken_args:
                    taken_args.add(short_cut[0])
                    args_alies[k].append(short_cut[0])
        for k, v in args_alies.items():
            args_alies[k] = [
                '-' * ((len(ii) > 1) + 1) + ii
                for ii in v
            ]
        logger.info(args_alies)
        for k, field in args_map.items():
            argument_parameters = dict(
                dest=k,
                action=ArgParseAction
            )
            if hasattr(field, '_get_arg_parse_kwargs'):
                argument_parameters.update(field._get_arg_parse_kwargs())
            parser.add_argument(
                *args_alies[k],
                **argument_parameters,
            )
        args = parser.parse_args(args, namespace)
        if hasattr(args, '_dict_storage'):
            for k, v in args._dict_storage.items():
                args_map[k]._set_value(v)

    def dump_yaml(self, **kwargs):
        if yaml is None:
            raise Exception(f'Please install pyyaml to use this function')
        return yaml.safe_dump(self._to_dict(), **kwargs)

    def dump_json(self, **kwargs):
        return json.dumps(self._to_dict(), **kwargs)


class ListValueField(ValueField, ArgParseField):
    """
        element_func returns an standard element.
    """

    def __init__(self, element=None, **kwargs):
        self.fields = []
        if element is None:
            element = Value('')
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

    def _get_arg_parse_kwargs(self):
        return dict(nargs='*')

    def _set_value(self, value):
        self._from_dict(value)


class MappingValueField(ValueField, ArgParseField):
    def __init__(self, default=None, map_dict=None):
        self.value = default
        self.map_dict = map_dict

    def _get_bin_value(self):
        return self.map_dict[self.value]

    def _set_value(self, value):
        assert type(value) is str
        self.value = value


class NodeMeta(type):
    def __new__(cls, clsname, superclasses, attributedict, **kwargs):
        logger.info(attributedict)
        attributedict['build'] = lambda: make_root_node(
            superclasses[0]._field, attributedict)
        attributedict['_new_field'] = lambda parameter_list: expression
        klass = type.__new__(cls, clsname, superclasses, attributedict)
        logger.info(klass)
        return klass


class Node(object, metaclass=NodeMeta):
    def __init__(self, *args, **kwargs):
        logger.info('init node')
        self.args = args
        self.kwargs = kwargs


class Config(Node):
    _field = ConfigField


class Value(Node):
    _field = ValueField


class ValueList(Node):
    _field = ListValueField
    _argparse_param = dict(nargs='*')


class ValueMap(Node):
    _field = MappingValueField
    _argparse_param = dict(nargs="*", action=None)


def make_node(node_instance, attribute_dict, arg_prefix=''):
    if arg_prefix != '':
        arg_prefix += '.'
    logger.info('node')
    for name, ndef in attribute_dict.items():
        if not isinstance(ndef, Node):
            continue
        logger.info(f'new node {name}')
        new_node = ndef._field(*ndef.args, **ndef.kwargs)
        if isinstance(new_node, ArgParseField):
            new_node._set_arg_parse_param_name(arg_prefix + name)
        make_node(new_node, ndef.__class__.__dict__, arg_prefix + name)
        logger.info(f'setting node {name}, {type(new_node)}')
        setattr(node_instance, name, new_node)
    return node_instance


def make_root_node(field_type, attribute_dict):
    node = field_type()
    return make_node(node, attribute_dict, 'cfg')
