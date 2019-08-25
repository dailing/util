from util.config import *
from random import randint
import time
from io import StringIO


def get_nested_configure():
    cfg = Configure()
    cfg.add('a', 0).add('b', 1)
    cfg.add_subconfigure('c').add('d', 2)
    return cfg


def test_add_get():
    cfg = Configure()
    cfg.add('a', 1).add('b', 2)
    cfg.\
        add_subconfigure('fuck').\
        add('a', 3)
    cfg.add_alias('alias', 'fuck.a')
    assert cfg.fuck.a == 3
    assert cfg.alias == 3
    cfg.fuck.a = 1000
    assert cfg.fuck.a == 1000, cfg.alias
    assert cfg._cfg_storage['fuck']._cfg_storage['a'] is \
        cfg._cfg_storage['alias']
    assert cfg.alias == 1000, cfg.alias


def test_from_dict():
    cfg = Configure()
    cfg.add('a', 0).add('b', 1)
    cfg.add_subconfigure('c').add('d', 2)
    dd = cfg.to_dict()
    assert dd['a'] == 0
    assert dd['b'] == 1
    assert dd['c']['d'] == 2

    cfg.from_dict(dict(a=100, b=101, c=dict(d=102)))
    cfg.a = 100
    cfg.b = 101
    cfg.c.d = 102
    dd = cfg.to_dict()
    assert dd['a'] == 100
    assert dd['b'] == 101
    assert dd['c']['d'] == 102


def test_mapping_field():
    cfg = Configure()
    cfg.add_mapping('a', dict(
        value1=1,
        value2=2,
    ), default_value='value1')
    assert cfg.a == 1
    cfg.a = 'value2'
    assert cfg.a == 2


def test_yaml():
    cfg = get_nested_configure()
    yaml_content = """
a: 100
b: 101
c:
  d: 102"""

    fake_file = StringIO(yaml_content)
    cfg.from_yaml(fake_file)

    assert cfg.a == 100
    assert cfg.b == 101
    assert cfg.c.d == 102

    new_yaml = cfg.to_yaml()
    fake_file = StringIO(new_yaml)
    cfg = get_nested_configure()
    cfg.from_yaml(fake_file)
    assert cfg.a == 100
    assert cfg.b == 101
    assert cfg.c.d == 102
