from os.path import exists


def check_exist(file):
    return file if exists(file) else None


def assert_exist(file):
    assert exists(file), f'File: [{file}] not exist'
    return file
