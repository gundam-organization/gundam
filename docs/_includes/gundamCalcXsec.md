### Description 

The `gundamCalcXsec` app is an interface to the `FitterEngine` which is in charge of performing a Cross Section calculation by taking the output of `gundamFitter` as input. 

## Usage

Similarly to [gundamFitter](gundamFitter.md), `gundamCalcXsec` takes a config file as input with `-c` but in addition it also takes the output of `gundamFitter` as an input with the `-f` argument.
```bash
gundamCalcXsec -c path/to/xsecconfig.yaml -f path/to/outputfit.root - path/to/output.root -t 12
```
Where `-o` and `-t` are the arguments for the output file path and the number of parallel threads respectively. 

In addition one can use `-n` and `-s` to specify the number of toys and to set a random seed respectively in the fit. 
```bash
gundamCalcXsec -c path/to/xsecconfig.yaml -f path/to/outputfit.root - path/to/output.root -t 12 -n 100 -s 5
```
### Trigger options

A certain number of trigger options can be used depending  on what the user needs:

| Option          | Description                                          |
| --------------- | ---------------------------------------------------- |
| -d              | Only overrides fitter config and prints it.          |
| -use-bf-as-xsec | Use best-fit as x-sec value instead of mean of toys. |
| --use-prefit    | Use prefit covariance matrices for the toy throws.   |

For a complete list of options run command without arguments.

### Config options

| Option                                                 | Type         | Description                                                     | Default |
|--------------------------------------------------------|--------------|-----------------------------------------------------------------|---------|
| [fitterEngineConfig](../configuration/FitterEngine.md) | json         | FitterEngine config                                             |         |
| minGundamVersion                                       | string       | gundamFitter will stop if the version is lower                  |         |
| outputFolder                                           | string       | Folder where the output file is written                         | ./      |
