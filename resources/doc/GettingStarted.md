## Getting started

### Applications

`GUNDAM` has a series of applications at its disposal:
- [gundamFitter](applications/gundamFitter.md)
- [gundamCalcXsec](applications/gundamCalcXsec.md)
- [gundamConfigCompare](applications/gundamConfigCompare.md)
- [gundamConfigUnfolder](applications/gundamConfigUnfolder.md)
- [gundamFitCompare](applications/gundamFitCompare.md)
- [gundamFitPlot](gundamFitPlot.md)
- [gundamFitReader](gundamFitReader.md)
- [gundamInputZipper](gundamInputZipper.md)
- [gundamPlotExtractor](gundamPlotExtractor.md)
- [gundamRoot](gundamRoot.md)
In general if you run the command without any arguments you get an explanation of what it does and what arguments can be used. For example if we take `gundamCalcXsec`:
```bash
[gundamCalcXsec.cxx]: Usage: 
[gundamCalcXsec.cxx]: ──────────── Main options: ────────────
[gundamCalcXsec.cxx]: configFile {-c,--config-file}: Specify path to the fitter config file (expected: 1 value)
[gundamCalcXsec.cxx]: fitterFile {-f}: Specify the fitter output file (expected: 1 value)
[gundamCalcXsec.cxx]: outputFile {-o,--out-file}: Specify the CalcXsec output file (expected: 1 value)
[gundamCalcXsec.cxx]: nbThreads {-t,--nb-threads}: Specify nb of parallel threads (expected: 1 value)
[gundamCalcXsec.cxx]: nToys {-n}: Specify number of toys (expected: 1 value)
[gundamCalcXsec.cxx]: randomSeed {-s,--seed}: Set random seed (expected: 1 value)
[gundamCalcXsec.cxx]: ──────────── Trigger options: ────────────
[gundamCalcXsec.cxx]: dryRun {-d,--dry-run}: Only overrides fitter config and print it. (trigger)
[gundamCalcXsec.cxx]: useBfAsXsec {--use-bf-as-xsec}: Use best-fit as x-sec value instead of mean of toys. (trigger)
[gundamCalcXsec.cxx]: usePreFit {--use-prefit}: Use prefit covariance matrices for the toy throws. (trigger)
```
The only exception is `gundamRoot` that immediately launches an interactive session. Launch with the `-h`  argument to get helpful output.