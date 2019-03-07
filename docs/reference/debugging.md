## Debugging

At some point something will not go as expected, and a closer look at what is going on is necessary. Using CMake it is easy to switch between building in `RELEASE` and `DEBUG` mode. It is highly suggested to compile in `DEBUG` mode for in-depth investigations.

### Check Ouput

It is certainly not perfect, but the code writes a lot of output to `std::cout` and `std::cerr` about what it's trying to do or what is going on. Please scan the output to see if the code is attempting to do what is expected.

If using `less` to read the text output, the `-R` flag can be used to correctly render the colored text (e.g. `less -R fit.log`). Alternatively, the colored text can be disabled through CMake options.

To capture the output from `stdout` and `stderr`, run a command with `&>file_name.log` appended to the end to capture all printed output in `file_name.log`, which can be any file name or extension. Use `1>file_name.log` or `2>file_name.log` to capture only `stdout` and `stderr` respectively. To both capture the output in a file and print to the screen, use the `tee` command.

Examples:
```bash
#Capture all output in file. Nothing printed to screen.
$ xsllhFit -j /path/to/config.json &>all_fit_output.log
#Capture stdout in a file, everything printed to screen.
$ xsllhFit -j /path/to/config.json | tee stdout_fit_output.log
#Capture all output in file and print to screen.
$ xsllhFit -j /path/to/config.json 2>&1 | tee all_fit_output.log
```

### Using GDB

GDB is a very powerful tool for debugging code, and can (sometimes) easily show the cause of a given error or behavior. First, make sure the code is compiled in `DEBUG` mode to enable the debugging symbols.

To run the fit (or any executable) in GDB, call GDB as follows:
```bash
$ gdb executable_name
(gdb) run <executable flags/options>

#Example running the fit executable.
$ gdb xsllhFit
(gdb) run -j /path/to/config.json
```

+ [GDB User Manual](http://sourceware.org/gdb/current/onlinedocs/gdb/)
+ [GDB Command Reference](https://darkdust.net/files/GDB%20Cheat%20Sheet.pdf)

### Using Google

Try Googling the error message. Try Googling how to use GDB. Always try to Google something. Learn some [Google Search Operators](https://ahrefs.com/blog/google-advanced-search-operators/) (aka Google-Fu).
