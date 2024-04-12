## gundamFitCompare
[< back to parent (GettingStarted)](../GettingStarted.md)
### Description 

The `gundamFitCompare` app that takes as input the output of [gundamFitter](gundamFitter.md) to produce comparison plots of the respective fit results. 

### Usage

One has to use the arguments `-f1` and `-f2` to specify the input fit files. Then using `-n1` and `-n2` names the fits respectively and lastly using `-a1` and `-a2` chooses the algorithm to compare (Migrad, Hesse, etc).
```bash
gundamFitCompare -f1 path/to/fit1.root -f2 path/to/fit2.root -n1 fit1 -n2 fit2 -a1 Hesse -a2 Hesse -o path/to/compare.root
```
The argument `-o` specifiers where the output compare file is saved. 

### Trigger options

A certain number of trigger options can be used depending  on what the user needs:

| Option     | Description                      |
| ---------- | -------------------------------- |
| --prefit-1 | Use prefit data only for file 1. |
| --prefit-2 | Use prefit data only for file 2. |
| --tree     | Comparing loaded events in tree. |
| -v         | Recursive verbosity printout.    |