## gundamConfigCompare
[< back to parent (GettingStarted)](../GettingStarted.md)
### Description 

The `gundamConfigCompare` app compares the elements in two config files. This app will output the differences it finds in the respective files and print out warnings if certain elements are present in one and not the other.

### Usage

It takes the path to each config file as arguments as follows:
```bash
gundamConfigCompare -c1 path/to/config1.yaml -c2 path/to/config2.yaml
```
Config files can either be in `.yaml` or `.json` format and used interchangeably. 

Override files can also be provided as inputs:
```bash 
gundamConfigCompare -c1 path/to/config1.yaml -c2 path/to/config2.yaml -of1 path/to/override1.yaml -of2 path/to/override2.yaml 
```

Additionally one can use the `-a` argument to show all the keys being compared.