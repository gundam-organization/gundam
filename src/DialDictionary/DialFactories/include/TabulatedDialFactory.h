#ifndef TABULATED_FACTORY_H_SEEN
#define TABULATED_FACTORY_H_SEEN
#include <ConfigUtils.h>
#include <DialCollection.h>

#include "DialFactoryBase.h"

class Event;
class DialBase;

/// Initialize the Tabulated type.  That yaml for this is
/// tableConfig:
///    - name: <table-name>                A table name passed to the library.
///      bins:  <number>                   The (optional) expected number of
///                                           bins in the table. Will be
///                                           overridden by the initFunction.
///      libraryPath: <path-to-library>    Location of the library
///      initFunction: <init-func-name>    Function called for initialization
///      initArguments: [<arg1>, ...]      List of argument strings (e.g.
///                                           input file names)
///      updateFunction: <update-func-name> Function called to update table
///      binningFunction: <bin-func-name>  Function to find bin index
///      binningVariables: [<var1>, ... ] Variables used for binning the
///                                           table "X" coordinate by the
///                                           binning function.
///
/// INITIALIZATION FUNCTION: If initFunction is provided it will be called with
/// the signature:
//
///    extern "C"
///    int initFunc(const char* name, int argc, const char* argv[], int bins)
///
///        name -- The name of the table
///        argc -- number of arguments
///        argv -- argument strings.  The arguments are defined by the library,
///                but are usually things like input file names for the lookup
///                table information.
///        bins -- The size of the table.  This is the expected size of the
///                table. The library must
///                choose an appropriate binning
//
/// The function should return less than or equal to zero for failure, and
/// otherwise, the number of elements needed to store the table (usually, the
/// same as the input value of "bins").
//
/// UPDATE TABLE FUNCTION: The updateFunction signature is:
///
///    extern "C"
///    int updateFunc(const char* name,
///                   double table[], int bins,
///                   const double par[], int npar)
///
///        name  -- table name
///        table -- address of the table to update
///        bins  -- The size of the table
///        par   -- The parameters.  Must match parameters
///                   define in the dial definition
///        npar  -- number of parameters
//
/// The function should return 0 for success, and any other value for failure
///
/// The table will be filled with "bins" values calculated with uniform
/// spacing between "low" and "high".  If bins is one, there must be one
/// value calculated for "low", if bins is two or more, then the first point
/// is located at "low", and the last point is located at "high".  The step
/// between the bins is (high-low)/(bins-1).  Examples:
///
///    bins = 2, low = 1.0, high = 6.0
///       values calculated at 1.0 and 6.0
//
///    bins = 3, low = 1.0, high = 6.0
///       values calculated at 1.0, 3.5, and 6.0
//
/// BIN INDEX DETERMINATION: The binningFunction signature is:

///    extern "C"
///    double binFunc(const char* name, int nvar, const double varv[], int bins);
///        name -- table name
///        nvar -- number of (truth) variables used to find bin
///        varv -- array of (truth) variables used to find bin
///        bins -- The number of bins in the table.
//
/// The function should return a double giving the fractional bin number
/// greater or equal to zero and LESS THAN the maximum number of bins.  The
/// integer part determines the index of the value below the value to be
/// interpolated, and the fractional part determines the interpolation between
/// the indexed bin, and the next.  This determines where to find the entry for
/// the input (truth) variables.
///
/// The code should loadable by dlopen, so it needs be compiled with (at least)
///
/// gcc -fPIC -rdynamic --shared -o <LibraryName>.so <source>
///
class TabulatedDialFactory : public DialFactoryBase {
public:
    TabulatedDialFactory() = default;
    ~TabulatedDialFactory() = default;
    TabulatedDialFactory(const ConfigReader& config_);

    /// Create an event-by-event weighting dial for this table.
    [[nodiscard]] DialBase* makeDial(const Event& event);

    /// Update the tabulated buffer.
    void updateTable(DialInputBuffer& inputBuffer);

    /// Get the table name
    const std::string& getName() {return _name_;}

    /// Get the path to the library that implements the tabulation.
    const std::string& getLibraryPath() {return _libraryPath_;}

    /// Get the name of the function that is called to initialize the
    /// tabulation.  The function is called during construction.
    const std::string& getInitializationFunction() {return _initFuncName_;}

    /// Get the arguments provided to the initialization function.
    const std::vector<std::string>& getInitializationArguments() {return _initArguments_;}

    /// Get the name of the function that determines the bin based on the
    /// event parameters
    const std::string& getBinningFunction() {return _binningFuncName_;}

    /// Get the name of the function to update the table.
    const std::string& getUpdateFunction() {return _updateFuncName_;}

    /// Get a vector of event variable names that are used to find the bin
    /// in the table for the event.
    const std::vector<std::string>& getBinningVariables() {return _binningVariableNames_;}

private:
    std::string _name_{"table"};
    std::string _libraryPath_;

    // The name of a symbol in the library that will be used to initialize the
    // library.
    std::string _initFuncName_;

    // The function that is attached to _initFuncName_.  It is called to
    // initialize the library.  See the class documentation above for a
    // description of what the library needs to implement.
    int (*_initFunc_)(const char* name,
                      int argc, const char* argv[],
                      int bins);

    /// A vector of arguments that will be passed to the initialization
    /// function.
    std::vector<std::string> _initArguments_;

    // The name of a symbol in the library that will be used to update the
    // table before each likelihood calculation.
    std::string _updateFuncName_;

    // The function that is attached to _updateFuncName_ symbol.
    int (*_updateFunc_)(const char* name,
                        double table[], int bins,
                        const double par[], int npar);

    // A cache to hold the table of calculated values
    std::vector<double> _table_;

    // The name of a symbol in the library that will be used to find the
    // bin associated with each event.
    std::string _binningFuncName_;

    // The function that is attached to _binningFuncName_ symbol.  See the
    // class documentation for what needs to be implemented.  The varv[] array
    // holds the event variables that are being binned.  The return value is a
    // double where the integer part is the bin number, and the fraction is
    // used to interpolated between bins.
    double (*_binningFunc_)(const char* name,
                            int varc, double varv[],
                            int bins);
    std::vector<std::string> _binningVariableNames_;
    std::vector<double> _binningVariableCache_;
};
#endif
