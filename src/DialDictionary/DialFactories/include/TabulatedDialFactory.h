/// @file TabulatedDialFactory.h
/// @brief Declares TabulatedDialFactory, which creates Tabulated dials that
///        interpolate in a lookup table filled by a user-supplied library.
/// @ingroup DialDictionary

#ifndef TABULATED_FACTORY_H_SEEN
#define TABULATED_FACTORY_H_SEEN
#include <ConfigUtils.h>
#include <DialCollection.h>

#include "DialFactoryBase.h"

class Event;
class DialBase;

/// Creates one Tabulated dial per event from a lookup table that a
/// user-supplied shared library refills for each set of parameter values.
///
/// A dial is a response function that turns fit-parameter values into an
/// event weight. For `dialType: Tabulated`,
/// DialCollection::initializeDialsWithTabulation() builds this factory from
/// the dial definition. The factory opens the library named in `tableConfig`
/// with `dlopen()`, looks up three functions in it with `dlsym()`, and owns
/// the table they work on:
/// - the *init* function, called once by the constructor, sets the table size;
/// - the *update* function refills the table for the current parameter
///   values. DialCollection registers updateTable() as an update callback
///   (DialCollection::addUpdate()), so it runs at every
///   DialManager::updateDialState(), before the events are reweighted;
/// - the *binning* function, called by makeDial() while the data are loaded,
///   turns the event's binning variables into a fractional table position.
///   The Tabulated dial stores that position once and, at each evaluation,
///   interpolates between two consecutive table entries.
///
/// @par Ownership
/// The DialCollection owns the factory (see DialFactoryBase). The factory owns
/// the table. Every Tabulated dial it creates keeps a raw pointer to that
/// table, and so does the GPU cache (Cache::Weight::Tabulated copies the table
/// at each reweight), so the factory must outlive its dials. The library is
/// never closed.
///
/// @par Thread safety
/// makeDial() is called from several DataDispenser threads. It and
/// updateTable() lock GundamGlobals::getGlobalMutEx(), since the library is
/// not assumed to be thread-safe.
///
/// @par Configuration and library functions
/// Initialize the Tabulated type.  That yaml for this is
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
///      binningFunction: <bin-func-name>  Function to find bin index
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
///        bins -- The size of the table.  This is the expected size of the
///                table. The library must
///                choose an appropriate binning
/// @endcode
///
/// The function should return less than or equal to zero for failure, and
/// otherwise, the number of elements needed to store the table (usually, the
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
/// The table will be filled with "bins" values calculated with uniform
/// spacing between "low" and "high".  If bins is one, there must be one
/// value calculated for "low", if bins is two or more, then the first point
/// is located at "low", and the last point is located at "high".  The step
/// between the bins is (high-low)/(bins-1).  Examples:
///
/// @code{.unparsed}
///    bins = 2, low = 1.0, high = 6.0
///       values calculated at 1.0 and 6.0
///
///    bins = 3, low = 1.0, high = 6.0
///       values calculated at 1.0, 3.5, and 6.0
/// @endcode
///
/// BIN INDEX DETERMINATION: The binningFunction signature is:
///
/// @code{.unparsed}
///    extern "C"
///    double binFunc(const char* name, int nvar, const double varv[], int bins);
///        name -- table name
///        nvar -- number of (truth) variables used to find bin
///        varv -- array of (truth) variables used to find bin
///        bins -- The number of bins in the table.
/// @endcode
///
/// The function should return a double giving the fractional bin number
/// greater or equal to zero and LESS THAN the maximum number of bins.  The
/// integer part determines the index of the value below the value to be
/// interpolated, and the fractional part determines the interpolation between
/// the indexed bin, and the next.  This determines where to find the entry for
/// the input (truth) variables.
///
/// The code should loadable by dlopen, so it needs be compiled with (at least)
///
/// @code{.unparsed}
/// gcc -fPIC -rdynamic --shared -o <LibraryName>.so <source>
/// @endcode
///
/// @note What the code does now, where it differs from the text above:
/// - `tableConfig` is read as a single mapping (keys directly under it, as in
///   `tests/fast-tests/200TabulatedAsimov-config.yaml`), not as a list.
/// - Every key except `bins` is mandatory, including `initFunction` and
///   `initArguments`: ConfigReader::checkConfiguration() stops the job if one
///   is missing. An empty `initFunction` string skips the init call.
/// - When `bins` is not set, the init function receives -1.
/// - The return value of the update function is ignored (see updateTable()).
/// - The factory has no `low` or `high` setting: where the table entries sit
///   is up to the library, which must keep its update and binning functions
///   consistent.
/// - A one-entry table does not work: a Tabulated dial always reads two
///   entries, at the stored index and the next one (see makeDial()).
///
/// @see DialFactoryBase, Tabulated, KrigedDialFactory
/// @ingroup DialDictionary
class TabulatedDialFactory : public DialFactoryBase {
public:
    /// Creates an unconfigured factory. Not used by GUNDAM, which always
    /// builds the factory from a configuration.
    /// @warning The library function pointers stay uninitialised, so
    ///          makeDial() and updateTable() must not be called on such an
    ///          object.
    TabulatedDialFactory() = default;

    /// Destroys the factory and its table. The library opened by the
    /// constructor is not closed.
    ~TabulatedDialFactory() = default;

    /// Reads `tableConfig`, loads the library, calls its init function and
    /// allocates the table.
    ///
    /// Environment variables are expanded in `libraryPath` and in each
    /// `initArguments` entry. The table gets the size returned by the init
    /// function (or the `bins` value when no init function is named) and is
    /// filled with NaN until the first updateTable().
    ///
    /// @param config_ The dial definition (the block holding
    ///        `dialType: Tabulated`), which must contain `tableConfig`.
    /// @warning A missing library, a missing symbol or an init return value
    ///          below 1 ends the process with `std::exit(EXIT_FAILURE)`
    ///          instead of throwing. A missing `tableConfig` throws
    ///          `std::runtime_error`.
    /// @warning If `initFunction` is empty and `bins` is not set, the table
    ///          size is -1, which `std::vector::resize()` cannot honour.
    TabulatedDialFactory(const ConfigReader& config_);

    /// Create an event-by-event weighting dial for this table.
    ///
    /// Reads the binning variables of `event` (getBinningVariables(), in that
    /// order) as doubles and passes them to the binning function. The integer
    /// part `i` of the result is clamped to `[0, size-2]` and the fractional
    /// part `f`, taken after that clamp, to `[0, 1]`; both are stored in the
    /// new Tabulated dial.
    ///
    /// @param event Event to build the dial for; it must hold every binning
    ///        variable.
    /// @return A new Tabulated dial owned by the caller, or `nullptr` if the
    ///         binning function returned a negative value (the event then
    ///         gets no dial from this collection).
    /// @warning Tabulated::evalResponse() returns
    ///          `f*table[i] + (1-f)*table[i+1]`: `f = 0` gives `table[i+1]`
    ///          and `f` close to 1 gives `table[i]`. This is the opposite
    ///          weighting to standard linear interpolation, which gives
    ///          `table[i]` at `f = 0`. Cache::Weight::Tabulated uses the same
    ///          formula as the CPU dial.
    [[nodiscard]] DialBase* makeDial(const Event& event);

    /// Update the tabulated buffer.
    ///
    /// Calls the library's update function on the table with the current
    /// input-parameter values of the collection. Called through the update
    /// callback registered by DialCollection::initializeDialsWithTabulation(),
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

    /// Get the path to the library that implements the tabulation.
    /// @return The `libraryPath` key, before environment-variable expansion.
    const std::string& getLibraryPath() {return _libraryPath_;}

    /// Get the name of the function that is called to initialize the
    /// tabulation.  The function is called during construction.
    /// @return The `initFunction` key (may be empty).
    const std::string& getInitializationFunction() {return _initFuncName_;}

    /// Get the arguments provided to the initialization function.
    /// @return The `initArguments` key, before environment-variable expansion.
    const std::vector<std::string>& getInitializationArguments() {return _initArguments_;}

    /// Get the name of the function that determines the bin based on the
    /// event parameters
    /// @return The `binningFunction` key.
    const std::string& getBinningFunction() {return _binningFuncName_;}

    /// Get the name of the function to update the table.
    /// @return The `updateFunction` key.
    const std::string& getUpdateFunction() {return _updateFuncName_;}

    /// Get a vector of event variable names that are used to find the bin
    /// in the table for the event.
    ///
    /// DialCollection::initializeDialsWithTabulation() adds each of them with
    /// DialCollection::addExtraLeafName(), so DataDispenser loads them.
    /// @return The `binningVariables` key, in configuration order.
    const std::vector<std::string>& getBinningVariables() {return _binningVariableNames_;}

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
    /// pointer that each Tabulated dial and Cache::Weight::Tabulated keep to
    /// it stays valid. Holds NaN until the first updateTable().
    std::vector<double> _table_;

    /// The name of a symbol in the library that will be used to find the
    /// bin associated with each event.
    std::string _binningFuncName_;

    /// The function that is attached to _binningFuncName_ symbol.  See the
    /// class documentation for what needs to be implemented.  The varv[] array
    /// holds the event variables that are being binned.  The return value is a
    /// double where the integer part is the bin number, and the fraction is
    /// used to interpolated between bins.
    double (*_binningFunc_)(const char* name,
                            int varc, double varv[],
                            int bins);

    /// Names of the event variables passed to the binning function
    /// (`binningVariables` key), in order.
    std::vector<std::string> _binningVariableNames_;

    /// Scratch buffer that makeDial() fills with the current event's binning
    /// variable values and passes as `varv`. One entry per binning variable;
    /// shared between calls, which the global mutex in makeDial() protects.
    std::vector<double> _binningVariableCache_;
};
#endif
