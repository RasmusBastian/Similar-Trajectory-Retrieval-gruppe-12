#!/usr/bin/env python3
"""
Two small edits to a fresh TSMini checkout (run from inside the TSMini folder):

 1. config.py: register the 'rome' dataset (prefix 'rome_20200'), so
    `--dataset rome` finds data/rome_20200.h5.
 2. utils/tool_funcs.py: make the NVIDIA GPU monitoring optional. The original
    calls nvmlInit() at import time, which crashes on a machine with no NVIDIA
    driver (e.g. WSL without a GPU).

Safe to run more than once.
"""
import pathlib

def edit(path, old, new, marker):
    p = pathlib.Path(path)
    s = p.read_text()
    if marker in s:
        print(f"{path}: already patched")
        return
    if old not in s:
        raise SystemExit(f"{path}: expected text not found, repo may have changed")
    p.write_text(s.replace(old, new, 1))
    print(f"{path}: patched")

edit(
    "config.py",
    "        elif 'xian' == cls.dataset:\n            cls.dataset_prefix = 'xian_20200'\n",
    "        elif 'xian' == cls.dataset:\n            cls.dataset_prefix = 'xian_20200'\n"
    "        elif 'rome' == cls.dataset:\n            cls.dataset_prefix = 'rome_20200'\n",
    "'rome' == cls.dataset",
)

edit(
    "utils/tool_funcs.py",
    "nvmlInit() # need initializztion here\n",
    "try:\n    nvmlInit() # need initializztion here\n    _NVML_OK = True\nexcept Exception: # no NVIDIA driver / GPU\n    _NVML_OK = False\n",
    "_NVML_OK",
)

edit(
    "utils/tool_funcs.py",
    "class GPUInfo:\n    _h = nvmlDeviceGetHandleByIndex(0)\n\n    @classmethod\n    def mem(cls):\n"
    "        info = nvmlDeviceGetMemoryInfo(cls._h)\n",
    "class GPUInfo:\n    _h = nvmlDeviceGetHandleByIndex(0) if _NVML_OK else None\n\n    @classmethod\n    def mem(cls):\n"
    "        if cls._h is None:\n            return 0, 0\n        info = nvmlDeviceGetMemoryInfo(cls._h)\n",
    "cls._h is None",
)
