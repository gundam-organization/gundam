#!/usr/bin/env python3
"""Profile CPU propagation with one thread for weights, caches and histograms.

One dataset load, a primed response cache and a constant Asimov point. Optional
perf sampling runs separately after the ordinary timing and counter passes.
"""
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    'scaling', Path(__file__).with_name('223CpuWeightScalingBenchmark.py'))
scaling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaling)

if __name__ == '__main__':
    scaling.main(single_thread=True, description=__doc__, benchmark_name=Path(__file__).stem)
