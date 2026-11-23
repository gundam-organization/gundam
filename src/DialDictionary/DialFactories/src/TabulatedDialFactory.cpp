/// @file TabulatedDialFactory.cpp
/// @brief Implements TabulatedDialFactory: loads the user table library and
///        creates one Tabulated dial per event.
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in TabulatedDialFactory.h.
/// The comments here describe how each method does its job.

#include "TabulatedDialFactory.h"

#include <GundamGlobals.h>
#include <DialBase.h>
#include <Tabulated.h>
#include <Event.h>

#include <GenericToolbox.Json.h>
#include <Logger.h>

#include <dlfcn.h>


/// @par Implementation
/// The keys are declared with ConfigReader::defineFields() and checked by
/// ConfigReader::checkConfiguration(), which ends the job when a mandatory key
/// is missing. The library is opened with `RTLD_LAZY` and its handle is a
/// local variable, so it is never closed. The global mutex is taken just
/// before `dlopen()` and held to the end of the constructor, so the init
/// function never runs at the same time as another call into a library.
/// Every later error calls `std::exit(EXIT_FAILURE)` rather than throwing.
/// The table is filled with NaN so that a use before the first updateTable()
/// shows up (NaN weights on the CPU; Cache::Weight::Tabulated::Apply() throws
/// on a non-finite entry).
TabulatedDialFactory::TabulatedDialFactory(const ConfigReader& config_) {

    // mandatory
    auto tableConfig = config_.fetchValue<ConfigReader>("tableConfig");

    tableConfig.defineFields({
        {FieldFlag::MANDATORY, "name"},
        {FieldFlag::MANDATORY, "libraryPath"},
        {FieldFlag::MANDATORY, "initFunction"},
        {FieldFlag::MANDATORY, "updateFunction"},
        {FieldFlag::MANDATORY, "binningFunction"},
        {FieldFlag::MANDATORY, "initArguments"},
        {FieldFlag::MANDATORY, "binningVariables"},
        {"bins"},
    });
    tableConfig.checkConfiguration();

    // mandatory options
    _name_ =  tableConfig.fetchValue<std::string>("name");
    _libraryPath_ =  tableConfig.fetchValue<std::string>("libraryPath");
    _initFuncName_ = tableConfig.fetchValue<std::string>("initFunction");
    _updateFuncName_ = tableConfig.fetchValue<std::string>("updateFunction");
    _binningFuncName_ = tableConfig.fetchValue<std::string>("binningFunction");
    _initArguments_ = tableConfig.fetchValue<std::vector<std::string>>("initArguments");
    _binningVariableNames_ = tableConfig.fetchValue<std::vector<std::string>>("binningVariables");

    // optional
    int bins =  tableConfig.fetchValue<int>("bins", -1); // table size: -1 if not set, then replaced by the init function's result

    _binningVariableCache_.resize(_binningVariableNames_.size());

    std::string expandedPath = GenericToolbox::expandEnvironmentVariables(getLibraryPath());

    LogInfo << "Create table: " << getName() << std::endl;
    LogInfo << "  Library path: " << getLibraryPath() << std::endl;
    LogInfo << "  Initialization function: " << getInitializationFunction() << std::endl;
    LogInfo << "  Update table function:   " << getUpdateFunction() << std::endl;
    LogInfo << "  Bin events function:     " << getBinningFunction() << std::endl;
    {
        int i{0}; // variable number shown in the log
        for (const std::string& var: getBinningVariables()) {
            LogInfo << "      Variable[" << i++ << "]: " << var << std::endl;
        }
    }

    if( not GenericToolbox::isFile(expandedPath) ){
        LogError << "File not found: " << getLibraryPath() << std::endl;
        LogError << "     Full path: " << expandedPath << std::endl;
        std::exit(EXIT_FAILURE); // Exit, not throw!
    }

    std::lock_guard<std::mutex> guard(GundamGlobals::getGlobalMutEx());
    void* library = dlopen(expandedPath.c_str(), RTLD_LAZY ); // library handle; never closed
    if( library == nullptr ){
        LogError << "Cannot load library: " << dlerror() << std::endl;
        std::exit(EXIT_FAILURE); // Exit, not throw!
    }

    // Get the initialization function.
    if (not getInitializationFunction().empty()) {
        void* initFunc = dlsym(library, getInitializationFunction().c_str());
        if( initFunc == nullptr ){
            LogError << "Initialization function symbol not found: "
                     << getInitializationFunction()
                     << std::endl;
            std::exit(EXIT_FAILURE); // Exit, not throw!
        }
        std::vector<std::string> argv_buffer; // expanded arguments; owns the strings argv points to
        for (std::string arg : getInitializationArguments()) {
            argv_buffer.push_back(GenericToolbox::expandEnvironmentVariables(arg));
        }
        std::vector<const char*> argv; // C-style argument array for the init function
        for (std::string& arg : argv_buffer) argv.push_back(arg.c_str());
        _initFunc_
            = reinterpret_cast<
                int(*)(const char* name,int argc, const char* argv[], int bins)
            >(initFunc);
        int result = _initFunc_(_name_.c_str(),(int) argv.size(), argv.data(), bins); // table size, or < 1 on failure
        if (result < 1) {
            LogError << "Error calling initialization function: "
                     << result
                     << std::endl;
            std::exit(EXIT_FAILURE); // Exit, not throw!
        }
        bins = result;
    }
    _table_.resize(bins);

    // Make sure things are going to fail badly if the table is used before
    // it is filled.
    for (auto& t : _table_) t = std::nan("not-set"); // one table entry

    // Get the update function
    void* updateFunc = dlsym(library, getUpdateFunction().c_str());
    if( updateFunc == nullptr ){
        LogError << "Update function symbol not found: "
                 << getUpdateFunction()
                 << std::endl;
        std::exit(EXIT_FAILURE); // Exit, not throw!
    }
    _updateFunc_
        = reinterpret_cast<
            int(*)(const char* name, double table[],
                   int bins, const double par[], int npar)>(updateFunc);

    // Get the binning function
    void* binningFunc = dlsym(library, getBinningFunction().c_str());
    if( binningFunc == nullptr ){
        LogError << "Binning function symbol not found: "
                 << getBinningFunction()
                 << std::endl;
        std::exit(EXIT_FAILURE); // Exit, not throw!
    }
    _binningFunc_
        = reinterpret_cast<
            double(*)(const char* name,
                   int varc, double varv[], int bins)>(binningFunc);
}

/// @par Implementation
/// Holds the global mutex while the library runs. The table and the parameter
/// values are passed as pointers to the vectors' storage, so the library
/// writes the table in place. The return value is discarded.
void TabulatedDialFactory::updateTable(DialInputBuffer& inputBuffer) {
    std::lock_guard<std::mutex> guard(GundamGlobals::getGlobalMutEx());
    _updateFunc_(_name_.c_str(),
                 _table_.data(),
                 (int) _table_.size(),
                 inputBuffer.getInputBuffer().data(),
                 (int) inputBuffer.getInputBuffer().size());
}

/// @par Implementation
/// Holds the global mutex for the whole call, because `_binningVariableCache_`
/// is shared between calls and the library is not assumed to be thread-safe.
/// The upper clamp compares `iBin` with `_table_.size()-2` in unsigned
/// arithmetic: for a table of fewer than two entries that value wraps around,
/// so no clamping happens. The debug printout is compiled only when
/// `TABULATED_DIAL_FACTORY_DUMP` is defined, which nothing in the repository
/// does.
DialBase* TabulatedDialFactory::makeDial(const Event& event) {
    std::lock_guard<std::mutex> guard(GundamGlobals::getGlobalMutEx());
    int i=0; // next slot of _binningVariableCache_
    for (const std::string& varName : getBinningVariables()) {
        double v = event.getVariables().fetchVariable(varName).getVarAsDouble(); // value of this binning variable
        _binningVariableCache_[i++] = v;
    }
    double bin = _binningFunc_(getName().c_str(), // fractional table position; negative means no dial
                               (int) _binningVariableCache_.size(),
                               _binningVariableCache_.data(),
                               (int) _table_.size());

    if (bin < 0.0) return nullptr;

    // Determine the bin index and the fractional part of the bin.
    int iBin = bin;
    if (iBin < 0) iBin = 0;     // Shouldn't happen, but just in case.
    if (iBin > _table_.size()-2) iBin = _table_.size()-2;
    double fracBin = bin - iBin;
    if (fracBin < 0.0) fracBin = 0.0;
    if (fracBin > 1.0) fracBin = 1.0;

#ifdef TABULATED_DIAL_FACTORY_DUMP
    // Summarize which table is being applied to the event.  This can be quite
    // useful for debugging parameter configurations, but is not compiled by
    // default since it will dominate the debug output.  We need to implement
    // different debug levels, and conditions.
    if (GundamGlobals::isDebug()) {
        std::ostringstream out;
        out << getName()
            << " " << iBin
            << " " << fracBin;
        std::size_t j = 0; // variable number shown in the printout
        for (const std::string& varName : getBinningVariables()) {
            out << " [" << j
                << "] " << varName
                << "=" << _binningVariableCache_[j];
            ++j;
        }
        out << std::endl;
        LogDebug << out.str() << std::endl;
    }
#endif

    // Do the unique_ptr dance in case there are exceptions.
    std::unique_ptr<Tabulated> dialBase = std::make_unique<Tabulated>(&_table_, iBin, fracBin);
    return dialBase.release();
}
