---
layout: default
title: gundamPlotExtractor
---

### Description 

The `gundamPlotExtractor` app extracts plots from the output of [gundamFitter](gundamFitter.md) into a designated diretory. 

### Usage

The user can pass the fit file with the `-f` argument alongside `-o` to define the output folder.
```bash
gundamPlotExtractor -f path/to/fit.root -c output/directory/
```

By default plots are saved as `.pdf` but one can define the extension with `-x` (ex. png, jpeg, svg).
```bash
gundamPlotExtractor -f path/to/fit.root -c output/directory/ -x png
```