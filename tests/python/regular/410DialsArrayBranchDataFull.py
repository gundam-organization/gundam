#!/usr/bin/env python3
"""Complementary full coverage for 410DialsArrayBranchData."""
from importlib import import_module
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fast'))

if __name__ == '__main__':
    sys.exit(import_module('410DialsArrayBranchData').main(full=True))
