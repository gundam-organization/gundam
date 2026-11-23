/// @file KrigedDialFactory.h
/// @brief Declares KrigedDialFactory, which creates Kriged dials that take a
///        weighted sum of entries of a table filled by a user-supplied library.
/// @ingroup DialDictionary

#ifndef KRIGED_FACTORY_H_SEEN
#define KRIGED_FACTORY_H_SEEN
#include <ConfigUtils.h>
#include <DialCollection.h>

#include "DialFactoryBase.h"

class Event;
class DialBase;

/// Creates one Kriged dial per event: a fixed list of table indices and
/// weights into a table that a user-supplied shared library refills for each
/// set of parameter values.
///
/// A dial is a response function that turns fit-parameter values into an
/// event weight. For `dialType: Kriged`,
/// DialCollection::initializeDialsWithKriging() builds this factory from the
/// dial definition. The factory opens the library named in `tableConfig` with
/// `dlopen()`, looks up three functions in it with `dlsym()`, and owns the
/// table they work on:
/// - the *init* function, called once by the constructor, sets the table size;
/// - the *update* function refills the table for the current parameter
///   values. DialCollection registers updateTable() as an update callback
///   (DialCollection::addUpdate()), so it runs at every
///   DialManager::updateDialState(), before the events are reweighted;
/// - the *weighting* function, called by makeDial() while the data are loaded,
///   gives the table indices and weights for the event. The Kriged dial
///   stores them once and, at each evaluation, returns the weighted sum of
///   the current table entries.
///
/// @par Ownership
/// The DialCollection owns the factory (see DialFactoryBase). The factory owns
/// the table. Every Kriged dial it creates keeps a raw pointer to that table,
/// and so does the GPU cache (Cache::Weight::Kriged), so the factory must
/// outlive its dials. The library is never closed.
///
/// @par Thread safety
/// makeDial() is called from several DataDispenser threads. It and
/// updateTable() lock GundamGlobals::getGlobalMutEx(), since the library is
/// not assumed to be thread-safe.
///
/// @par Configuration and library functions
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
/// @code{.yaml}
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
/// @endcode
///
/// INITIALIZATION FUNCTION: If initFunction is provided it will be called with
/// the signature:
///
/// @code{.unparsed}
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
/// @endcode
///
/// The function should return less than or equal to zero for failure, and
/// otherwise, the number of elements needed to store the table (Often the
/// same as the input value of "bins").
///
/// UPDATE TABLE FUNCTION: The updateFunction signature is:
///
/// @code{.unparsed}
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
/// @endcode
///
/// The function should return 0 for success, and any other value for failure
///
/// EVENT WEIGHT DETERMINATION: The weightingFunction signature is:
///
/// @code{.unparsed}
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
/// @endcode
///
/// The function fills arrays of table index, and the weight for the index
/// that should be used to calculate the weight for an event. The function
/// should return the number of elements filled in the index and weight
/// arrays.  The dial value for an event is then calculated as the sum of
/// weight[i]*table[index[i]].
///
/// The code should loadable by dlopen, so it needs be compiled with (at least)
///
/// @code{.unparsed}
/// gcc -fPIC -rdynamic --shared -o <LibraryName>.so <source>
/// @endcode
///
/// @note What the code does now, where it differs from the text above:
/// - The keys are `weightFunction` (not `weightingFunction`) and
///   `weightVariables` (not `binningVariables`), as in
///   `tests/fast-tests/200KrigedAsimov-config.yaml`. With the names above, the
///   job stops on a missing mandatory key.
/// - `tableConfig` is read as a single mapping, not as a list.
/// - Every key except `bins`, `maxWeights` and `weightNormalization` is
///   mandatory, including `initFunction` and `initArguments`. An empty
///   `initFunction` string skips the init call. When `bins` is not set, the
///   init function receives -1.
/// - Two optional keys are not listed above: `maxWeights` (the `maxEntries`
///   size of the index and weight arrays; values below 10000 are raised to
///   10000) and `weightNormalization` (expected sum of an event's weights,
///   default 1; zero or negative disables the check done in makeDial()).
/// - The weighting function is called with seven arguments,
///   `(name, bins, nvar, varv, maxEntries, index, weights)`: the trailing
///   `int bins` in the signature above is not passed, and `varv` is passed as
///   a non-const `double*`.
/// - The return value of the update function is ignored (see updateTable()).
///
/// @see DialFactoryBase, Kriged, TabulatedDialFactory
/// @ingroup DialDictionary
class KrigedDialFactory : public DialFactoryBase {
public:
    /// Creates an unconfigured factory. Not used by GUNDAM, which always
    /// builds the factory from a configuration.
    /// @warning The library function pointers stay uninitialised, so
    ///          makeDial() and updateTable() must not be called on such an
    ///          object.
    KrigedDialFactory() = default;

    /// Destroys the factory and its table. The library opened by the
    /// constructor is not closed.
    ~KrigedDialFactory() = default;

    /// Reads `tableConfig`, loads the library, calls its init function and
    /// allocates the table.
    ///
    /// Environment variables are expanded in `libraryPath` and in each
    /// `initArguments` entry. The table gets the size returned by the init
    /// function (or the `bins` value when no init function is named) and is
    /// filled with NaN until the first updateTable().
    ///
    /// @param config_ The dial definition (the block holding
    ///        `dialType: Kriged`), which must contain `tableConfig`.
    /// @warning A missing library, a missing symbol or an init return value
    ///          below 1 ends the process with `std::exit(EXIT_FAILURE)`
    ///          instead of throwing. A missing `tableConfig` throws (LogThrowIf).
    /// @warning If `initFunction` is empty and `bins` is not set, the table
    ///          size is -1, which `std::vector::resize()` cannot honour.
    KrigedDialFactory(const ConfigReader& config_);

    /// Create an event-by-event weighting dial for this table.
    ///
    /// Reads the weight variables of `event` (getWeightVariables(), in that
    /// order) as doubles and calls the weighting function, which fills up to
    /// `maxEntries` table indices and weights. The new Kriged dial copies the
    /// filled pairs, storing the weights as `float`.
    ///
    /// @param event Event to build the dial for; it must hold every weight
    ///        variable.
    /// @return A new Kriged dial owned by the caller, or `nullptr` if the
    ///         weighting function returned fewer than one entry (the event
    ///         then gets no dial from this collection).
    /// @note When `weightNormalization` is positive and the weights do not sum
    ///       to it within a relative 1e-6, a warning and the event summary are
    ///       logged; the dial is still created.
    /// @warning The returned entry count is not checked against `maxEntries`;
    ///          a library that returns more reads past the index and weight
    ///          buffers.
    [[nodiscard]] DialBase* makeDial(const Event& event) override;

    /// Update the kriged buffer.
    ///
    /// Calls the library's update function on the table with the current
    /// input-parameter values of the collection. Called through the update
    /// callback registered by DialCollection::initializeDialsWithKriging(),
    /// that is once per DialManager::updateDialState(), even when the
    /// parameters have not changed.
    ///
    /// @param inputBuffer The collection's first input buffer
    ///        (`DialCollection::getDialInputBufferList().front()`), already
    ///        refreshed by DialCollection::updateInputBuffers(). Its values
    ///        are passed as `par`, in buffer order (the `dialInputList` order
    ///        when that list is given).
    /// @warning The update function's return value is ignored, so a failure
    ///          reported by the library does not stop the fit.
    void updateTable(DialInputBuffer& inputBuffer);

    /// Get the table name
    /// @return The `name` key of `tableConfig`, passed to every library call.
    const std::string& getName() {return _name_;}

    /// Get the path to the library that implements the kriging.
    /// @return The `libraryPath` key, before environment-variable expansion.
    const std::string& getLibraryPath() {return _libraryPath_;}

    /// Get the name of the function that is called to initialize the
    /// kriging.  The function is called during construction.
    /// @return The `initFunction` key (may be empty).
    const std::string& getInitializationFunction() {return _initFuncName_;}

    /// Get the arguments provided to the initialization function.
    /// @return The `initArguments` key, before environment-variable expansion.
    const std::vector<std::string>& getInitializationArguments() {return _initArguments_;}

    /// Get the name of the function that determines the bin based on the
    /// event parameters
    /// @return The `weightFunction` key.
    /// @note It is the name of the weighting function, which gives the table
    ///       indices and weights of an event rather than a bin.
    const std::string& getWeightFunction() {return _weightFuncName_;}

    /// Get the name of the function to update the table.
    /// @return The `updateFunction` key.
    const std::string& getUpdateFunction() {return _updateFuncName_;}

    /// Get a vector of event variable names that are used to find the indices
    /// and weights in the table for the event.
    ///
    /// DialCollection::initializeDialsWithKriging() adds each of them with
    /// DialCollection::addExtraLeafName(), so DataDispenser loads them.
    /// @return The `weightVariables` key, in configuration order.
    const std::vector<std::string>& getWeightVariables() {return _weightVariableNames_;}

private:
    /// Table name (`name` key), passed as the first argument of every library
    /// call. The default is always replaced, since the key is mandatory.
    std::string _name_{"table"};

    /// Library path as written in the configuration (`libraryPath` key);
    /// environment variables are expanded only when the library is opened.
    std::string _libraryPath_;

    /// The name of a symbol in the library that will be used to initialize the
    /// library.
    std::string _initFuncName_;

    /// The function that is attached to _initFuncName_.  It is called to
    /// initialize the library.  See the class documentation above for a
    /// description of what the library needs to implement.
    ///
    /// Only called by the constructor; left uninitialised when no init
    /// function is named.
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
    /// Called by updateTable().
    int (*_updateFunc_)(const char* name,
                        double table[], int bins,
                        const double par[], int npar);

    /// A cache to hold the table of calculated values
    ///
    /// Sized once by the constructor and never resized afterwards, so the
    /// pointer that each Kriged dial and Cache::Weight::Kriged keep to it
    /// stays valid. Holds NaN until the first updateTable().
    std::vector<double> _table_;

    /// The name of a symbol in the library that will be used to find the
    /// table indices and weights associated with each event.
    std::string _weightFuncName_;

    /// The function that is attached to _weightFuncName_ symbol.  See the
    /// class documentation for what needs to be implemented.  The varv[] array
    /// holds the event variables that are being weighted.  The return value is
    /// a double where the integer part is the bin number, and the fraction is
    /// used to interpolated between bins.
    ///
    /// @note The return value is an `int`: the number of index/weight pairs
    ///       the function filled. makeDial() makes no dial when it is below 1.
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
    ///
    /// Read from the optional `maxWeights` key and raised to at least 10000
    /// by the constructor. makeDial() sizes `_indexCache_` and `_weightCache_`
    /// to it on first use and passes it to the library as `maxEntries`.
    int _minimumWeightSpace_{10000};

    /// The expected normalization of the weights.  Zero or negative to
    /// disable the check.
    ///
    /// Read from the optional `weightNormalization` key. makeDial() logs a
    /// warning when an event's weights sum to a value that differs from it by
    /// more than 1e-6 times this value.
    double _weightNormalization_{1.0};
};
#endif
