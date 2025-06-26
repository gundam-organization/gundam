#include "KrigedDialFactory.h"

#include <GundamGlobals.h>
#include <DialBase.h>
#include <Kriged.h>
#include <Event.h>

#include <GenericToolbox.Json.h>
#include <Logger.h>

#include <dlfcn.h>

KrigedDialFactory::KrigedDialFactory(const ConfigReader& config_){
    LogThrowIf(not config_.hasField("tableConfig"), "tableConfig must be defined in:" << config_.toString(true));

    auto tableConfig = config_.fetchValue<ConfigReader>("tableConfig");
    tableConfig.defineFields({
        {FieldFlag::MANDATORY, "name"},
        {FieldFlag::MANDATORY, "libraryPath"},
        {FieldFlag::MANDATORY, "initFunction"},
        {FieldFlag::MANDATORY, "initArguments"},
        {FieldFlag::MANDATORY, "updateFunction"},
        {FieldFlag::MANDATORY, "weightFunction"},
        {FieldFlag::MANDATORY, "weightVariables"},
        {"maxWeights"},
        {"weightNormalization"},
        {"bins"},
    });
    tableConfig.checkConfiguration();

    tableConfig.fillValue(_name_, "name");
    tableConfig.fillValue(_libraryPath_, "libraryPath");
    tableConfig.fillValue(_initFuncName_, "initFunction");
    tableConfig.fillValue(_initArguments_, "initArguments");
    tableConfig.fillValue(_updateFuncName_, "updateFunction");
    tableConfig.fillValue(_weightFuncName_, "weightFunction");
    tableConfig.fillValue(_weightVariableNames_, "weightVariables");
    tableConfig.fillValue(_minimumWeightSpace_, "maxWeights");
    tableConfig.fillValue(_weightNormalization_, "weightNormalization");
    int bins =  tableConfig.fetchValue<int>("bins", -1);

    _weightVariableCache_.resize(_weightVariableNames_.size());
    if (_minimumWeightSpace_ < 10000) _minimumWeightSpace_ = 10000;
    std::string expandedPath = GenericToolbox::expandEnvironmentVariables(getLibraryPath());

    LogInfo << "Create table: " << getName() << std::endl;
    LogInfo << "  Library path: " << getLibraryPath() << std::endl;
    LogInfo << "  Initialization function: " << getInitializationFunction() << std::endl;
    LogInfo << "  Update table function:   " << getUpdateFunction() << std::endl;
    LogInfo << "  Weight events function:     " << getWeightFunction() << std::endl;
    {
        int i{0};
        for (const std::string& var: getWeightVariables()) {
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
        std::lock_guard<std::mutex> guard(GundamGlobals::getGlobalMutEx());
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

    // Get the weight function
    void* weightFunc = dlsym(library, getWeightFunction().c_str());
    if( weightFunc == nullptr ){
        LogError << "Weight function symbol not found: "
                 << getWeightFunction()
                 << std::endl;
        std::exit(EXIT_FAILURE); // Exit, not throw!
    }
    _weightFunc_
        = reinterpret_cast<
            int(*)(const char* name, int bins,
                   int varc, double varv[],
                   int maxEntries,
                   int index[], double weight[])>(weightFunc);

}

void KrigedDialFactory::updateTable(DialInputBuffer& inputBuffer) {
    std::lock_guard<std::mutex> guard(GundamGlobals::getGlobalMutEx());
    _updateFunc_(_name_.c_str(),
                 _table_.data(),
                 (int) _table_.size(),
                 inputBuffer.getInputBuffer().data(),
                 (int) inputBuffer.getInputBuffer().size());
}

DialBase* KrigedDialFactory::makeDial(const Event& event) {
    std::lock_guard<std::mutex> guard(GundamGlobals::getGlobalMutEx());
    int i=0;
    for (const std::string& varName : getWeightVariables()) {
        double v = event.getVariables().fetchVariable(varName).getVarAsDouble();
        _weightVariableCache_[i++] = v;
    }
    if (_indexCache_.size() < _minimumWeightSpace_) {
        _indexCache_.resize(_minimumWeightSpace_);
    }
    if (_weightCache_.size() != _indexCache_.size()) {
        _weightCache_.resize(_indexCache_.size());
    }
    int entries = _weightFunc_(getName().c_str(), _table_.size(),
                               (int) _weightVariableCache_.size(),
                               _weightVariableCache_.data(),
                               (int) _indexCache_.size(),
                               _indexCache_.data(),
                               _weightCache_.data());

#undef KRIGED_DIAL_FACTORY_DUMP
#ifdef KRIGED_DIAL_FACTORY_DUMP
    // Summarize which table is being applied to the event.  This can be quite
    // useful for debugging parameter configurations, but is not compiled by
    // default since it will dominate the debug output.  We need to implement
    // different debug levels, and conditions.
    if (GundamGlobals::isDebug()) {
        std::ostringstream out;
        out << getName()
            << " entries: " << entries;
        std::size_t j = 0;
        for (const std::string& varName : getWeightVariables()) {
            out << " [" << j
                << "] " << varName
                << "=" << _weightVariableCache_[j];
            ++j;
        }
        out << std::endl;
        LogDebug << out.str() << std::endl;
    }
#endif
#undef KRIGED_DIAL_FACTORY_DUMP

    if (entries < 1) return nullptr;

    if (_weightNormalization_ > 0.0) {
        double sum = 0.0;
        for (int i = 0; i < entries; ++i) {
            sum += _weightCache_[i];
        }
        if (std::abs(sum-_weightNormalization_) > 1E-6*_weightNormalization_) {
            LogWarning << "Denormalized Kriging dial -- "
                       << " sum: " << sum
                       << " Normalization: " << _weightNormalization_
                       << " Difference: " << std::abs(sum-_weightNormalization_)
                       << std::endl;
            LogWarning << "Denormalized Event Summary: " << event << std::endl;
        }
    }

    // Do the unique_ptr dance in case there are exceptions.
    std::unique_ptr<Kriged> dialBase
        = std::make_unique<Kriged>(&_table_, entries, _indexCache_, _weightCache_);

    return dialBase.release();
}
