"""加载文件名以数字开头的 clean 模块。"""

import importlib.util
import sys
from pathlib import Path


def load_clean_module(filename, module_name):
    """按文件路径加载 clean 模块。"""
    path = Path(__file__).resolve().parent / filename
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
