#include "TabulatedDialFactory.h"

#include <GundamGlobals.h>
#include <DialBase.h>
#include <Tabulated.h>
#include <Event.h>

#include <GenericToolbox.Json.h>
#include <Logger.h>

#include <dlfcn.h>


TabulatedDialFactory::TabulatedDialFactory(const ConfigUtils::ConfigReader& config_) {

    // mandatory
    auto tableConfig = config_.fetchValue<ConfigReader>("tableConfig");

    // mandatory options
    _name_ =  tableConfig.fetchValue<std::string>("name");
    _libraryPath_ =  tableConfig.fetchValue<std::string>("libraryPath");
    _initFuncName_ = tableConfig.fetchValue<std::string>("initFunction");
    _updateFuncName_ = tableConfig.fetchValue<std::string>("updateFunction");
    _binningFuncName_ = tableConfig.fetchValue<std::string>("binningFunction");
    _initArguments_ = tableConfig.fetchValue<std::vector<std::string>>("initArguments");
    _binningVariableNames_ = tableConfig.fetchValue<std::vector<std::string>>("binningVariables");

    // optional
    int bins =  tableConfig.fetchValue<int>("bins", -1);

    _binningVariableCache_.resize(_binningVariableNames_.size());

    std::string expandedPath = GenericToolbox::expandEnvironmentVariables(getLibraryPath());

    LogInfo << "Create table: " << getName() << std::endl;
    LogInfo << "  Library path: " << getLibraryPath() << std::endl;
    LogInfo << "  Initialization function: " << getInitializationFunction() << std::endl;
    LogInfo << "  Update table function:   " << getUpdateFunction() << std::endl;
    LogInfo << "  Bin events function:     " << getBinningFunction() << std::endl;
    {
        int i{0};
        for (const std::string& var: getBinningVariables()) {
            LogInfo << "      Variable[" << i++ << "]: " << var << std::endl;
        }
    }

    if( not GenericToolbox::isFile(expandedPath) ){
        LogError << "File not found: " << getLibraryPath() << std::endl;
        LogError << "     Full path: " << expandedPath << std::endl;
        std::exit(EXIT_FAILURE); // Exit, not throw!
    }

    void* library = dlopen(expandedPath.c_str(), RTLD_LAZY );
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
        std::vector<std::string> argv_buffer;
        for (std::string arg : getInitializationArguments()) {
            argv_buffer.push_back(GenericToolbox::expandEnvironmentVariables(arg));
        }
        std::vector<const char*> argv;
        for (std::string& arg : argv_buffer) argv.push_back(arg.c_str());
        _initFunc_
            = reinterpret_cast<
                int(*)(const char* name,int argc, const char* argv[], int bins)
            >(initFunc);
        int result = _initFunc_(_name_.c_str(),(int) argv.size(), argv.data(), bins);
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
    for (auto& t : _table_) t = std::nan("not-set");

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

void TabulatedDialFactory::updateTable(DialInputBuffer& inputBuffer) {
    _updateFunc_(_name_.c_str(),
                 _table_.data(),
                 (int) _table_.size(),
                 inputBuffer.getInputBuffer().data(),
                 (int) inputBuffer.getInputBuffer().size());
}

DialBase* TabulatedDialFactory::makeDial(const Event& event) {
    int i=0;
    for (const std::string& varName : getBinningVariables()) {
        double v = event.getVariables().fetchVariable(varName).getVarAsDouble();
        _binningVariableCache_[i++] = v;
    }
    double bin = _binningFunc_(getName().c_str(),
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
        std::size_t j = 0;
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
