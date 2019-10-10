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
        self.__dict__['value'] = default
        self.__dict__['value_type'] = type(default)

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

    # def __getattr__(self, name):
    #     assert isinstance(self.value, dict)

    # def __setattr__(self, name, value):
    #     assert isinstance(self.value, dict)
    #     self.value[name] = value


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
        try:
            return self._cfg_storage[name]._get_bin_value()
        except KeyError:
            logger.error(
                f'{name} not an attribute: {list(self._cfg_storage.keys())}')
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

    def _from_dict(self, values):
        for k, v in values.items():
            logger.info(f'{self._node_name}. setting key {k}, value : {v}')
            if k not in self._cfg_storage:
                raise Exception(
                    f'key {k} not in the config defination!'
                    f'{self} ')
            self._cfg_storage[k]._from_dict(v)

    def _from_yaml(self, file):
        if type(file) is str:
            file = open(file, 'r')
        dd = yaml.safe_load(file)
        self._from_dict(file)

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
        args, rest_args = parser.parse_known_args(args, namespace)
        if hasattr(args, '_dict_storage'):
            for k, v in args._dict_storage.items():
                args_map[k]._set_value(v)
        while len(rest_args) > 0:
            key, v = rest_args[:2]
            rest_args = rest_args[2:]
            if not key.startswith('--'):
                continue
            key = key[2:].split('.')
            if not(key[0] == 'cfg'):
                continue
            mapping_key = key[-1]
            key = '.'.join(key[:-1])
            if key not in args_map:
                continue
            vv = args_map[key]
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
                pass
            vv.value[mapping_key] = v

    def dump_yaml(self, **kwargs):
        if yaml is None:
            raise Exception(f'Please install pyyaml to use this function')
        return yaml.safe_dump(self._to_dict(), **kwargs)

    def from_yaml(self, file):
        if type(file) is str:
            file = open(file, 'r')
        dd = yaml.safe_load(file)
        self._from_dict(dd)

    def dump_json(self, **kwargs):
        return json.dumps(self._to_dict(), **kwargs)


class ListValueField(ValueField, ArgParseField):
    """
        element_func returns an standard element.
    """

    def __init__(self, element=None, **kwargs):
        self.fields = []
        if element is None:
            logger.warning('element is None')
            element = Value('')
        self.element_func = element

    def append(self, value=None):
        logger.info(self.element_func)
        ele = self.element_func.build()
        logger.info(self.element_func)
        logger.info(list(ele.__class__.__dict__.keys()))
        logger.info(value)
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

    def _get_arg_parse_kwargs(self):
        return dict(nargs='*')

    def _set_value(self, value):
        self._from_dict(value)


class MappingValueField(ValueField, ArgParseField):
    def __init__(self, default=None, **kwargs):
        self.value = default
        self.map_dict = kwargs

    def _get_bin_value(self):
        return self.map_dict[self.value]

    def _set_value(self, value):
        assert type(value) is str, value
        self.value = value


class NodeMeta(type):
    def __new__(cls, clsname, superclasses, attributedict, **kwargs):
        logger.info(f'{clsname}: {attributedict}')
        logger.info(superclasses)
        field_type = None
        for i in superclasses[::-1]:
            if hasattr(i, '_field'):
                field_type = getattr(i, '_field')
        if '_field' in attributedict:
            field_type = attributedict['_field']
        attributedict['build'] = lambda: make_root_node(
            field_type, attributedict)
        # attributedict['_new_field'] = lambda parameter_list: expression
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
    node_instance.__dict__['_node_name'] = arg_prefix
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
