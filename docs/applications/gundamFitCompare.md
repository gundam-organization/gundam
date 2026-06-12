---
layout: default
title: gundamFitCompare
---

### Description 

The `gundamFitCompare` app that takes as input the output of [gundamFitter](gundamFitter.md) to produce comparison plots of the respective fit results. 

### Usage

It takes the path to each fit file as arguments as follows:
```bash
gundamConfigCompare -f1 path/to/fit1.root -c2 path/to/fit2.root
```
The output files path is specified by the argument `-o`:
```bash
gundamFitCompare -f1 path/to/fit1.root -f2 path/to/fit2.root -o path/to/compare.root
 ```

Additionally, the comparison algorithm (Migrad, Hesse, etc.) can be specified using `-a1` and `-a2`, and the fits can be named using `-n1` and `-n2`, respectively:
```bash
gundamFitCompare -f1 path/to/fit1.root -f2 path/to/fit2.root -n1 fit1 -n2 fit2 -a1 Hesse -a2 Hesse -o path/to/compare.root
``` 

### Trigger options

A certain number of trigger options can be used depending  on what the user needs:

| Option     | Description                      |
| ---------- | -------------------------------- |
| --prefit-1 | Use prefit data only for file 1. |
| --prefit-2 | Use prefit data only for file 2. |
| --tree     | Comparing loaded events in tree. |
| -v         | Recursive verbosity printout.    |