# AGENTS.md


## Project overview
GUNDAM is a C++ code which core functionality allows to propagate
parameters on histograms in order to compute a binned likelihood.


## Repository structure
- `src/`: C++ code
  - Each subfolder represents a library (except for `Applications` 
  which gather executables source code)
  - Each library holds either a pair of `src` and `include` subfolders
  or a sub-library with its name, itself containing `src` and `include`.
- `tests/`: tests that ensure the code is producing accurate results.
- `cmake/`: `./CMakeLists.txt` refers to sub-CMake files within this directory.


## Architecture
Libraries are designed such that no cross-dependences are
made. From lower to higher abstraction level, libraries
are hierarchical as such:
- Utils
- ParametersManager (depends on Utils)
- SamplesManager (depends on ParametersManager)
- DialDictionary (depends on SamplesManager)
- CacheManager (depends on DialDictionary)
- Propagator (depends on DialDictionary and optionnaly CacheManager)
- DatasetManager (depends on Propagator)
- StatisticalInference (depends on Propagator and DatasetManager)
- Fitter (depends on StatisticalInference)

The `CMakeList.txt` of `Applications` attach the
highest level library the app will require.


## Coding policy

### Spelling

General spelling uses `PascalCase` for classes/structs/namespaces
and also file names, 
while all the rest is using `camelCase` (functions, methods, members...).

In order to make the code easier to read, we use additional
conventions.
For private class members, we chose to add `_` before the name.
For instance: `MyObject _myObject_`.

For objects passed in methods or functions, we only use
a trailing `_`: `double getParameter(int myIndex_)`.

Protected members, although rarely used, use a `_` as a prefix.
For instance: `MyProtectedObject _obj;`

For public member are only using their names writen in
camelCase: `MyPublicObject obj;`

### Headers include ordering

Always include higher level of abstraction includes
first. This allows to spot missing includes in dependencies
while compiling.
Typically, we use a 4 layer of includes:
```c++
// This project headers
#include "FitterEngine.h"
#include "ConfigUtils.h"

// submodules headers
#include "Logger.h"
#include "GenericToolbox.Utils.h"

// external libraries
#include <TFile.h>
#include <TH1D.h>

// std
#include <string>
#include <vector>
```

This way, if a header is missing in `FitterEngine.h`,
compiler will throw an error.


### Configuration based classes

GUNDAM is designed such that users would fill the
engine content on runtime with a set of configuration
files (JSON). 
Classes that are acting as a frontend to the
user via the configuration inherit from `JsonBaseClass`
that itself inherits from `ConfigClass` which provides
a comon set of methods to first `.configure()` (parse a config),
and then `.initialize()` to perform the appropriate data loading
if needed. Once initialized, a given class is considered
as fully loaded and can be used for its core 
functionalities.

In practice, introducing a new class based on `JsonBaseClass` 
means that we are supposed to implement the configure and
inialize routine via overriding `configureImpl()` and `initializeImpl()`.

Reading the configuration is handled by a `ConfigReader` object, owned by
the class itself as `_config_`.
The standard procedure to `configure()` our class is to first reset the `_config_`
object (for backward compatibility reasons) and then define the valid `fields` as:
```c++
_config_.clearFields();
_config_.defineFields({
  {"keyName"},
  {"myOptionKey", {"optionAltKeyName"}},
});
```

`keyName` directly refers to the key used in the config file (yaml or json), and
GUNDAM will be able to read its value.
In some cases, key names have evolved over time and to key backward compatibility
we use alternative names list which will redirect the correct action and print a
warning. In this example `myOptionKey` is the up-to-date key name, while this
field can also be recognized as `optionAltKeyName` in the JSON/YAML file.
For implementing new classes, alternative keys should not be used.

Important note: we often implement a key exactly called `name`. Its value isn't
always used in the code, but it's often useful to keep as the config override
logic features `name` as a way to identify a given entry in a list.

Then one should do the following that will ensure the user config is valid:
```c++
_config_.checkConfiguration();
```

Then use the key name (not the alternative names) to tell the config reader where
to save the value with `fillValue`.
```c++
_config_.fillValue(_myMember_, "keyName");
```

### Use of enums

Enums are very useful in coding to explicit intended values.
However, converting back and forth from string often leads to
repeated code semantics.
In order to prevent repeated code, I created a macro deployed with
`#include`. You can implement a new enum as such:
```c++
#define ENUM_NAME MyEnum
#define ENUM_FIELDS \
  ENUM_FIELD(Unset, -1) \
  ENUM_FIELD(Foo) \
  ENUM_FIELD(Bar)
#include "GenericToolbox.MakeEnum.h"
```

This last line deploys a simple `struct` that act like an `enum` but
with extra functionalities.
Here an example of handling those. Note that `MyEnum::` is expected.

```c++
MyEnum myEnum = MyEnum::Foo;
if( myEnum == MyEnum::Foo ){ /* do the Foo thing */ }
LogInfo << myEnum << std::endl; // std::cout stream works
std::string myEnumAsString = myEnum.toString(); // string conversions
myEnum = MyEnum::getEnumVal(0); // direct assignment with int
myEnum = MyEnum::toEnum("Foo"); // direct assignment with string
MyEnum::generateVectorStr(); // returns the list of the available enum names
```

Such `enum` can be autofilled with the config file as a string with:
```c++
_config_.fillEnum(myEnum, "enumAsString");
```


### On the importance of finding appropriate names

Class names are important to determine
the scope of functionality of a given object.
For instance:
- `SampleSet.cpp` holds a "set of samples" which materializes
  as a `std::vector<Sample>`. Its members refer to global
  options applying over each `Sample`.
- `Propagator` is an object that handles the propagation
  of parameters onto sample histograms. It owns a `SampleSet`
  and a bunch of parameters contained within a `ParametersManager`.
  It also includes a set of "dials" that tells how each
  parameter should propagate to the sample histograms.
  Since its name is `Propagator`, its core functionality is
  to "propagate". Thus, it contains `propagateParameters()`
  in its API. In a more global point of view, a `Propagator` in
  GUNDAM is defined for a `model` and for `data`. They are used 
  to compute a likelihood. Following `Propagator` name choice,
  there's no question of implementing likelihood related methods
  in its core.

Each class should have a well define name that would at first order
describe its purpose.
If its functionality is too high level (abstraction), then create another
class that will be owned by it.
This convention helps to provide clear code base for devs to check
the overall logic.


## Implementing tests

When one implement a new features, it is highly recommended to add
a dedicated test to ensure the feature is working as intended.
`./tests` holds different folders: `fast-tests`, `regular-tests`, `slow-tests` and
`extended-tests`.
All `fast-tests` are require to pass in order to validate the continuous
integration runners on GitHub.
I recommend to implement a `.py` test, handling the creation of the inputs
and the running of GUNDAM. 
Python tests allow to use the `PythonInterface` library of GUNDAM to
check the engine behavior while running.
For example bases, read `tests/fast-tests/21*.py` files.
GUNDAM tests are run using the dedicated bash script `tests/gundam-tests.sh`.
You can run a specific test using for instance:
```bash
cd ./tests
bash gundam-tests.sh -a -t fast-tests/210IgnoreZeroPredictionAtPrior.py -v
```

Note that in order to succeed, python tests must run with GUNDAM compiled using
the CMake option `-D WITH_PYTHON_INTERFACE=ON`. This requires `pybind11`
installed on the machine.
Also make sure that the GUNDAM python interface is recognizing the right
library while testing.
In non-local test-mode it uses `$PYTHONPATH` to grab the Python library of GUNDAM.


## Build commands

```bash
cd <this-repo-dir>
cmake -S . -B cmake-build-debug
cmake --build cmake-build-debug -j3
```

