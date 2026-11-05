#!/usr/bin/env python3
"""Compiled variable transforms populate the expected samples and histogram bins."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))
from EvalFromLib import (build_config_text, compile_shared_library, evaluate_config,
                         write_input_root_file, write_transform_source_file)


def main():
    work = Path.cwd()
    root = work / '210ApiEvalFromLib.root'
    source = work / '210ApiEvalFromLib.cpp'
    library = work / '210ApiEvalFromLib.so'
    write_input_root_file(root)
    write_transform_source_file(source)
    compile_shared_library(source, library)
    evaluate_config(build_config_text(root, library, 'new'), work, 'variableDict/evalFromLib')
    print('SUCCESS: compiled variable transforms populate the expected bins.')


if __name__ == '__main__':
    main()
