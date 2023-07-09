import json
import os
import sys
from pathlib import Path
from collections import OrderedDict

from datetime import datetime


def cmd_str():
    """ get cmd that starts current script"""
    args = ' '.join(sys.argv)
    return f'python {args}'


def datetime_str():
    dtstr = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    return dtstr


def instantiate(path, *args, **kwargs):
    """ instantiate a class or call a function with given args and kwargs 
        from `anywhere`.

        Example:
            `instantiate('module.a.B', t=1e-3)`
            return `module.a.B(t=1e-3)`
    """
    from pydoc import locate
    obj = locate(path)
    return obj(*args, **kwargs)


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def auto_rename(path, ignore_ext=False):
    count = 1
    new_path = path
    while True:
        if not os.path.exists(new_path):
            return new_path
        file_name = os.path.basename(path)
        try:
            name, ext = file_name.split('.')
            new_name = f'{name}_{count}.{ext}'
        except:
            new_name = f'{file_name}_{count}'
        if ignore_ext:
            new_name = f'{file_name}_{count}'
        new_path = os.path.join(os.path.dirname(path), new_name)
        count += 1
