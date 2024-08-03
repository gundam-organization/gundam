## Scripts

This directory contains a number of scripts written to help with the analysis in some way. Most scripts do a single task and are either BASH scripts or C++ based ROOT scripts.

The ROOT scripts have primarily been developed and tested with ROOT 6 compiled with C++11 standards. This may cause issues when running the scripts with ROOT 5 and/or ROOT compiled with C++03 or lower. The ROOT scripts are intended to be loaded in a ROOT interactive session using `.L script_name.cxx` then run using `script_name()` with appropriate parameters like a normal function.
