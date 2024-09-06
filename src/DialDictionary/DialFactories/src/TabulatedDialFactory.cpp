#include "TabulatedDialFactory.h"

#include <GundamGlobals.h>
#include <DialBase.h>
#include <Tabulated.h>
#include <Event.h>

#include <GenericToolbox.Json.h>
#include <Logger.h>

#include <dlfcn.h>

#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[Tabulated]"); });
#endif

TabulatedDialFactory::TabulatedDialFactory(const JsonType& config_) {
    auto tableConfig = GenericToolbox::Json::fetchValue<JsonType>(config_, "tableConfig");

    _name_ =  GenericToolbox::Json::fetchValue<std::string>(tableConfig, "name", _name_);

    _libraryPath_ =  GenericToolbox::Json::fetchValue<std::string>(tableConfig, "libraryPath");

    _initFuncName_ = GenericToolbox::Json::fetchValue<std::string>(tableConfig, "initFunction", _initFuncName_);
    _initArguments_ = GenericToolbox::Json::fetchValue(tableConfig, "initArguments", _initArguments_);

    _updateFuncName_ = GenericToolbox::Json::fetchValue<std::string>(tableConfig, "updateFunction");

    _binningFuncName_ = GenericToolbox::Json::fetchValue<std::string>(tableConfig, "binningFunction");

    int bins =  GenericToolbox::Json::fetchValue<int>(tableConfig, "bins", -1);

    _binningVariableNames_ = GenericToolbox::Json::fetchValue(tableConfig, "binningVariables", _binningVariableNames_);
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
    if (inputBuffer.isMasked()) return;
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
    if (iBin > _table_.size()-1) iBin = _table_.size()-1;
    double fracBin = bin - iBin;
    if (fracBin < 0.0) fracBin = 0.0;
    if (fracBin > 1.0) fracBin = 1.0;

    // Do the unique_ptr dance in case there are exceptions.
    std::unique_ptr<Tabulated> dialBase = std::make_unique<Tabulated>(&_table_, iBin, fracBin);
    return dialBase.release();
}
