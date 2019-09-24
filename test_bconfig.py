from util.bconfig import *
from random import randint
import time
from io import StringIO


class ConfigNested2(Config):
    v1 = Value(30)


class ConfigNested1(Config):
    v1 = Value(20)
    nc2 = ConfigNested2()


class ConfigDef(Config):
    v1 = Value(10)
    v2 = Value('v2_test_value1')
    nc1 = ConfigNested1()
    vlist = ValueList(Value)
    vmapping = ValueMap('mv1', dict(
        mv1=lambda x: x * x,
        mv2=1000
    ))


def test_get_value():
    cfg = ConfigDef.build()
    assert cfg.v1 == 10
    assert cfg.v2 == 'v2_test_value1'
    assert cfg.nc1.v1 == 20
    assert cfg.nc1.nc2.v1 == 30


def test_set_value():
    cfg = ConfigDef.build()
    cfg.v1 = 100
    cfg.v2 = 'asdfasf'
    cfg.nc1.v1 = 200
    cfg.nc1.nc2.v1 = 300
    assert cfg.v1 == 100
    assert cfg.v2 == 'asdfasf'
    assert cfg.nc1.v1 == 200
    assert cfg.nc1.nc2.v1 == 300
    t2_instance = ConfigDef.build()
    assert t2_instance.v1 == 10
    assert t2_instance.v2 == 'v2_test_value1'
    assert t2_instance.nc1.v1 == 20
    assert t2_instance.nc1.nc2.v1 == 30


def test_get_dict():
    cfg = ConfigDef.build()
    dd = cfg._to_dict()
    assert dd['v1'] == 10
    assert dd['v2'] == 'v2_test_value1'
    assert dd['nc1']['v1'] == 20
    assert dd['nc1']['nc2']['v1'] == 30


def test_list_value():
    cfg = ConfigDef.build()
    values = [10, 11, 'fuck']
    for i in values:
        cfg.vlist.append(i)
    for i, j, k in zip(range(len(values)), values, cfg.vlist):
        assert cfg.vlist[i] == j
        assert j == k


def test_mapping_value():
    cfg = ConfigDef.build()
    assert cfg.vmapping(10) == 100
    assert cfg.vmapping(11) == 121
    cfg.vmapping = 'mv2'
    assert cfg.vmapping == 1000


def test_to_yaml():
    cfg = ConfigDef.build()
    try:
        yaml_content = cfg.dump_yaml()
        logger.info(yaml_content)
    except Exception as e:
        assert False


def test_type_convert():
    cfg = ConfigDef.build()
    cfg.v1 = '412123'
    assert cfg.v1 == 412123


def test_parse_arg():
    cfg = ConfigDef.build()
    cfg.parse_args('--cfg.v1 1000 --v2 override_val'.split(), )
    assert cfg.v1 == 1000
    assert cfg.v2 == 'override_val'


def test_parse_list():
    cfg = ConfigDef.build()
    values = [10, 30, 40, 'adf', '234']
    cfg.parse_args(('--vlist '+' '.join(map(str, values))).split())
    for i, j in zip(cfg.vlist, values):
        assert str(i) == str(j)
    assert len(cfg.vlist) == len(values)


def test_parse_mapping():
    cfg = ConfigDef.build()
    cfg.parse_args('--vmapping mv2'.split())
    assert cfg.vmapping == 1000
    cfg.parse_args('--vmapping mv1'.split())
    assert cfg.vmapping(11) == 121
