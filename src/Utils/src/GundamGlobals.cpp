//
// Created by Nadrino on 11/02/2021.
//

#include "GundamGlobals.h"
#include "Logger.h"

#include "TRandom3.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[GlobalVariables]");
});

// INIT
bool GundamGlobals::_enableDevMode_{false};
bool GundamGlobals::_disableDialCache_{false};
bool GundamGlobals::_enableCacheManager_{false};
bool GundamGlobals::_lightOutputMode_{false};
int GundamGlobals::_nbThreads_ = 1;
std::mutex GundamGlobals::_threadMutex_;
std::map<std::string, bool> GundamGlobals::_boolMap_;
std::vector<TChain*> GundamGlobals::_chainList_;
VerboseLevel GundamGlobals::_verboseLevel_{NORMAL_MODE};
GenericToolbox::ParallelWorker GundamGlobals::_threadPool_;


void GundamGlobals::setVerboseLevel(VerboseLevel verboseLevel_){
  _verboseLevel_ = verboseLevel_;
  LogWarning << "Verbose level set to: " << VerboseLevelEnumNamespace::toString(_verboseLevel_) << std::endl;
}
void GundamGlobals::setVerboseLevel(int verboseLevel_){ GundamGlobals::setVerboseLevel(static_cast<VerboseLevel>(verboseLevel_)); }
void GundamGlobals::setNbThreads(int nbThreads_){
  _nbThreads_ = nbThreads_;
  _threadPool_.reset();
  _threadPool_.setCheckHardwareCurrency(false);
  _threadPool_.setNThreads(_nbThreads_);
  _threadPool_.setCpuTimeSaverIsEnabled(true);
  _threadPool_.initialize();
}

void GundamGlobals::setEnableCacheManager(bool enable) { _enableCacheManager_ = enable;}
void GundamGlobals::setDisableDialCache(bool disableDialCache_){ _disableDialCache_ = disableDialCache_; }
void GundamGlobals::setLightOutputMode(bool enable_){ _lightOutputMode_ = enable_; }
bool GundamGlobals::getEnableCacheManager() {return _enableCacheManager_;}
bool GundamGlobals::isDisableDialCache(){ return _disableDialCache_; }
bool GundamGlobals::isLightOutputMode(){ return _lightOutputMode_; }
bool GundamGlobals::isEnableDevMode(){ return _enableDevMode_; }
VerboseLevel GundamGlobals::getVerboseLevel(){ return _verboseLevel_; }
const int& GundamGlobals::getNbThreads(){ return _nbThreads_; }
std::mutex& GundamGlobals::getThreadMutex() { return _threadMutex_; }
std::map<std::string, bool>& GundamGlobals::getBoolMap() { return _boolMap_; }
std::vector<TChain*>& GundamGlobals::getChainList() { return _chainList_; }
GenericToolbox::ParallelWorker &GundamGlobals::getParallelWorker() {
  return _threadPool_;
}
