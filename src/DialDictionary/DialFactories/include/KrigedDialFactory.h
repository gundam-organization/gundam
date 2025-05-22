#ifndef KRIGED_FACTORY_H_SEEN
#define KRIGED_FACTORY_H_SEEN
#include <ConfigUtils.h>
#include <DialCollection.h>

#include "DialFactoryBase.h"

class Event;
class DialBase;

/// Create a kriged dial.  This can be used for any event weighting that can
/// be expressed as the sum of weight[i]*table[index[i]] where the table is
/// filled for each new set of fit parameters, while the weight and index
/// arrays are precalculated (and constant for a particular event).  This is
/// mostly intended to use for multi-dimensional functions that can be
/// tabulated (like the atmospheric neutrino oscillation weights which is
/// indexed by neutrino energy and zenith angle) and well approximated using
/// linear kriging.  It can also be used to implement linear interpolation
/// into a table (for one or more dimensions).  The contrast between a Kriged
/// and a Tabulated dial is that a tabulated dial does linear interpolation
/// between two sequential entries in a table (so it can be used for long
/// baseline neutrino oscillation weights indexed by energy), and the kriged
/// dial can be used for tables with any number of dimensions, but at the cost
/// of using more memory and less computational efficiency.
///
/// The yaml for this is
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
///      weightingFunction: <bin-func-name>  Find the weights for an event
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
///        bins -- The size of the table.  This is the suggested size of the
///                table, but can be ignored. The library must choose an
///                appropriate binning, and returns the actual number of
///                elements.
//
/// The function should return less than or equal to zero for failure, and
/// otherwise, the number of elements needed to store the table (Often the
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
/// EVENT WEIGHT DETERMINATION: The weightingFunction signature is:
///
///    extern "C"
///    int weightTable(const char* name, int bins,
///                    int nvar, const double varv[],
///                    int maxEntries, int index[], double weights[],
///                    int bins);
///        name       -- table name
///        bins       -- The number of bins in the table
///        nvar       -- number of (truth) variables provided.
///        varv[]     -- array of (truth) variables provided.
///        maxEntries -- The number of elements allocated in the index
///                and weights arrays.
///        index[]    -- (output) array of indices to weight for an event.
///        weight[]   -- (output) array of weights for an event.
///
/// The function fills arrays of table index, and the weight for the index
/// that should be used to calculate the weight for an event. The function
/// should return the number of elements filled in the index and weight
/// arrays.  The dial value for an event is then calculated as the sum of
/// weight[i]*table[index[i]].
///
/// The code should loadable by dlopen, so it needs be compiled with (at least)
///
/// gcc -fPIC -rdynamic --shared -o <LibraryName>.so <source>
///
class KrigedDialFactory : public DialFactoryBase {
public:
    KrigedDialFactory() = default;
    ~KrigedDialFactory() = default;
    KrigedDialFactory(const JsonType& config_);

    /// Create an event-by-event weighting dial for this table.
    [[nodiscard]] DialBase* makeDial(const Event& event) override;

    /// Update the kriged buffer.
    void updateTable(DialInputBuffer& inputBuffer);

    /// Get the table name
    const std::string& getName() {return _name_;}

    /// Get the path to the library that implements the kriging.
    const std::string& getLibraryPath() {return _libraryPath_;}

    /// Get the name of the function that is called to initialize the
    /// kriging.  The function is called during construction.
    const std::string& getInitializationFunction() {return _initFuncName_;}

    /// Get the arguments provided to the initialization function.
    const std::vector<std::string>& getInitializationArguments() {return _initArguments_;}

    /// Get the name of the function that determines the bin based on the
    /// event parameters
    const std::string& getWeightFunction() {return _weightFuncName_;}

    /// Get the name of the function to update the table.
    const std::string& getUpdateFunction() {return _updateFuncName_;}

    /// Get a vector of event variable names that are used to find the indices
    /// and weights in the table for the event.
    const std::vector<std::string>& getWeightVariables() {return _weightVariableNames_;}

private:
    std::string _name_{"table"};
    std::string _libraryPath_;

    /// The name of a symbol in the library that will be used to initialize the
    /// library.
    std::string _initFuncName_;

    /// The function that is attached to _initFuncName_.  It is called to
    /// initialize the library.  See the class documentation above for a
    /// description of what the library needs to implement.
    int (*_initFunc_)(const char* name,
                      int argc, const char* argv[],
                      int bins);

    /// A vector of arguments that will be passed to the initialization
    /// function.
    std::vector<std::string> _initArguments_;

    /// The name of a symbol in the library that will be used to update the
    /// table before each likelihood calculation.
    std::string _updateFuncName_;

    /// The function that is attached to _updateFuncName_ symbol.
    int (*_updateFunc_)(const char* name,
                        double table[], int bins,
                        const double par[], int npar);

    /// A cache to hold the table of calculated values
    std::vector<double> _table_;

    /// The name of a symbol in the library that will be used to find the
    /// table indices and weights associated with each event.
    std::string _weightFuncName_;

    /// The function that is attached to _weightFuncName_ symbol.  See the
    /// class documentation for what needs to be implemented.  The varv[] array
    /// holds the event variables that are being weighted.  The return value is
    /// a double where the integer part is the bin number, and the fraction is
    /// used to interpolated between bins.
    int (*_weightFunc_)(const char* name, int bins,
                        int varc, double varv[],
                        int maxEntries, int index[], double weight[]);

    /// The names of the weight variable branches
    std::vector<std::string> _weightVariableNames_;

    /// The double values for the weight values (read from the branchs).  This
    /// is used to pass the values to the _weightFunc_
    std::vector<double> _weightVariableCache_;

    /// A pre-allocated cache for the _weightFunc_ index[] parameter.  The
    /// vector should be large enough to hold the largest possible number of
    /// weights returned by _weightFunc_ (the size is set by
    /// _minimumWeightSpace_).
    std::vector<int> _indexCache_;

    /// A pre-allocated cache for the _weightFunc_ weight[] parameter.  The
    /// vector should be large enough to hold the largest possible number of
    /// weights returned by _weightFunc_ (the size is set by
    /// _minimumWeightSpace_).
    std::vector<double> _weightCache_;

    /// The minimum number of weights that must be allowed by for an event.
    /// This should usually be a big number.
    int _minimumWeightSpace_{10000};

    /// The expected normalization of the weights.  Zero or negative to
    /// disable the check.
    double _weightNormalization_{1.0};
};
#endif
