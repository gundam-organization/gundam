## xsllhTreeConvert

This program provides a base for translating event trees into the format expected by the fit code. This has basic support for HighLAND2 files, but still may require analysis specific tweaks. The "flattree" for the fit contains a small set of variables for both selected and true events.

It is designed to be run with a JSON configuration file. The usage of the command is as follows:
```bash
$ xsllhTreeConvert -j </path/to/config.json>
```
The `-j` flag is required and is the config file for the fit. Currently there are no other options; all settings are specified in the configure file.

### Configuration file

Examples in the repository. Full description coming soon.

