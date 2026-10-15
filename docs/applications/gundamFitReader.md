---
layout: default
title: gundamFitReader
---

### Description 

The `gundamFitReader` app reads the fit output file from [gundamFitter](gundamFitter.md) and prints out relevant fit information. 
### Usage

Using the `-f` argument the user can specify the input fit file.
```bash
gundamFitReader -c path/to/fit.root 
```
Similarly using the `-e` trigger and the `-o` option, the output can be written in the a ROOT file.
```bash
gundamFitReader -c path/to/fit.root -e -o path/to/output.root
```
### Trigger options

A certain number of trigger options can be used depending  on what the user needs:

| Option          | Description                                        |
| --------------- | -------------------------------------------------- |
| -e              | Export data to output files                        |
| --show-par-list | Show parameters list  
| --show-par-list | Show parameters list |
| --compare       | Compare two fit files. Example: gundamFitCompare -f1 path/to/fit1.root -f2 path/to/fit2.root -n1 fit1 -n2 fit2 -a1 Hesse -a2 Hesse -o path/to/compare.root |
