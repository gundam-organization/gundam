---
layout: default
title: gundamInputZipper
---

### Description 

The `gundamInputZipper` app reads in the config inputs and copies the inputs into a unified ZIP file. 
### Usage

Using the `-c` and `-of` arguments one can point to the desired config:
```bash
gundamInputZipper -c path/to/config.yaml -of path/to/override.yaml
```
This will create a directory with the same name as the config file.

Otherwise one can specify the name of the output with `-o` 
```bash 
gundamInputZipper -c path/to/config.yaml -of path/to/override.yaml -o output/
```

In addition, the user can define the maximum size (in MB) of the file with `--max-size` and zip the output folder with `-z`:
```bash
gundamInputZipper -c path/to/config.yaml -of path/to/override.yaml -o output/ -z --max-size 50
```