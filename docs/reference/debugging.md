## Debugging

At some point something will not go as expected, and a closer look at what is going on is necessary. Using CMake it is easy to switch between building in `RELEASE` and `DEBUG` mode. It is highly suggested to compile in `DEBUG` mode for in-depth investigations.

### Check Ouput

It is certainly not perfect, but the code writes a lot of output to `std::cout` and `std::cerr` about what it's trying to do or what is going on. Please scan the output to see if the code is attempting to do what is expected.

If using `less` to read the text output, the `-R` flag can be used to correctly render the colored text (e.g. `less -R fit.log`). Alternatively, the colored text can be disabled through CMake options.

### Using GDB

GDB is a very powerful tool for debugging code, and can (sometimes) easily show the cause of a given error or behavior. First, make sure the code is compiled in `DEBUG` mode to enable the debugging symbols.

To run the fit (or any executable) in GDB, call GDB as follows:

```bash
$ gdb xsllhFit
(gdb) run -j /path/to/config.json
```
