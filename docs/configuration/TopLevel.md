---
layout: default
title: Top Level Configuration Options.
---
# Configuring GUNDAM

GUNDAM is configured using YAML (Json) input files that describe and control every aspect of the job.  For the most part, the input files are directly handled as standard YAML, with one minor exception to include files into the configuration.

## Best practices: Including other files into the configuration.

The recommended practice is to break the configuration into several input
files that reflect the logical structure and to include them using the
include file mechanism.

The GUNDAM libraries start by reading the entire configuration in to
memory, and then it is interpreted into a json structure according to the
[usual rules](https://www.yaml.org) with one exception.  If the value of a
string is found to end in the extension ".yaml", then the string is treated
as a filename relative to the current working directory to be included at
that point, and the contents of the file is substituted into the input
stream.  For instance

```yaml
fitterEngineConfig: "./inputs/engineConfiguration.yaml"
```

will read the contents of the "./inputs/engineConfiguration.yaml" file as a `Json` object add it to the full `Json` object as a child of the "fitterEngineConfig" field.  This provides convenient ways to factor the full configuration into components.

```yaml
minGundamVersion: 2.0.0
fitterEngineConfig:
    minimizerConfig:
        type: RootMinimizer
        minimizer: Minuit2
    likelihoodInterfaceConfig:
        jointPropabilityConfig:
            type: BarlowLLH
```
could be broken into three files:

- topConfig.yaml
```yaml
minGundamVersion: 2.0.0
fitterEngineConfig:
    minimizerConfig: "./minimizerConfig.yaml"
    likelihoodInterfaceConfig: "./likelihoodConfig.yaml"
```
  - minimizerConfig.yaml
```yaml
type: RootMinimizer
minimizer: Minuit2
```

  - likelihoodConfig.yaml
```yaml
jointPropabilityConfig:
    type: BarlowLLH
```

## Top Level Configuration Options.

| Option | Type | Description | Default |
|------------------------------------------|--------|------------------------------------------------|---------|
| [fitterEngineConfig](FitterEngine.md)    | json   | FitterEngine config                            |         |
| minGundamVersion (the format is "v.r.p") | string | gundamFitter will stop if the version is lower |         |
| outputFolder                             | string | Folder where the output file is written        | ./      |
