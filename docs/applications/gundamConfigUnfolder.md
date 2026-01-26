---
layout: default
title: gundamConfigUnfolder
---

### Description 

The `gundamConfigUnfolder` app takes a config file and and exports all related config files into one `.json` that can be used as an input for other apps that use said input.

### Usage

Using the `-c` argument one provides the `.yaml` or `.json` config file:
```bash
gundamConfigUnfolder -c path/to/config.yaml -o path/to/output.json
```

One can also include an override file with `-of` to include config override.
```bash
gundamConfigUnfolder -c path/to/config.yaml -of path/to/override.yaml -o path/to/output.json
```