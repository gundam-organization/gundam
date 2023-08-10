## PlotGenerator

[< back to parent (Propagator)](./Propagator.md)

### Description

### Config options
Note that the leaf used in the `varDictionnaries` must be an int.

### Example

```varDictionnaries:
  - name: "reaction"
    dictionary:
      - { value: -1, title: "no truth", color: 1 } # kBlack
      - { value: 0, title: "CCQE", color: 801 } # kOrange+1
      - { value: 1, title: "RES", color: 413 } # kGreen-3
      - { value: 2, title: "DIS", color: 843 } # kTeal+3
      - { value: 3, title: "COH", color: 867 } # kAzure+7
      - { value: 4, title: "NC", color: 430 } # kCyan-2
      - { value: 5, title: "CC-#bar{#nu}_{#mu}", color: 593 } # kBlue-7
      - { value: 6, title: "CC-#nu_{e}, CC-#bar{#nu}_{e}", color: 602 } # kBlue+2
      - { value: 7, title: "out FV", color: 909 } # kPink+9
      - { value: 9, title: "2p2h", color: 634 } # kRed+2
      - { value: 777, title: "sand #mu", color: 920 } # kGray
      - { value: 999, title: "other", color: 922 } # kGray+2

histogramsDefinition:
  - varToPlot: "Raw"

  - varToPlot: "Pmu"
    splitVars: ["", "reaction"]
    useSampleBinning: true # if not possible, error
    rescaleAsBinWidth: true # default true -> to look like a PDF
    rescaleBinFactor: 30 
    xMax: 3000.
    xTitle: "p_{#mu} (MeV)"
    yTitle: "Counts"

  - varToPlot: "CosThetamu"
    splitVars: ["", "reaction"]
    useSampleBinning: true # if not possible, error
    rescaleAsBinWidth: true # default true -> to look like a PDF
    rescaleBinFactor: 1 
    xMax: 1.
    xTitle: "cos#theta_{#mu}"
    yTitle: "Counts"

canvasParameters:
  height: 800
  width: 1200
  nbXplots: 3
  nbYplots: 2
```
