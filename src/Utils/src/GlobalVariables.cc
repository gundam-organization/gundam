//
// Created by Nadrino on 11/02/2021.
//

#include "GlobalVariables.h"
#include "Logger.h"

#include "TRandom3.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[GlobalVariables]");
});

// INIT
bool GlobalVariables::_debugVerbose_{false};
bool GlobalVariables::_enableDevMode_{false};
int GlobalVariables::_nbThreads_ = 1;
std::mutex GlobalVariables::_threadMutex_;
std::map<std::string, bool> GlobalVariables::_boolMap_;
std::vector<TChain*> GlobalVariables::_chainList_;
GenericToolbox::ParallelWorker GlobalVariables::_threadPool_;

bool GlobalVariables::_enableCacheManager_{false};


void GlobalVariables::setDebugVerbose(bool debugVerbose_){
  _debugVerbose_ = debugVerbose_;
}
void GlobalVariables::setNbThreads(int nbThreads_){
  _nbThreads_ = nbThreads_;
  _threadPool_.reset();
  _threadPool_.setCheckHardwareCurrency(false);
  _threadPool_.setNThreads(_nbThreads_);
  _threadPool_.setCpuTimeSaverIsEnabled(true);
  _threadPool_.initialize();
}

void GlobalVariables::setEnableCacheManager(bool enable) {_enableCacheManager_ = enable;}
bool GlobalVariables::getEnableCacheManager() {return _enableCacheManager_;}
bool GlobalVariables::isEnableDevMode(){ return _enableDevMode_; }
bool GlobalVariables::isDebugVerbose(){ return _debugVerbose_; }
const int& GlobalVariables::getNbThreads(){ return _nbThreads_; }
std::mutex& GlobalVariables::getThreadMutex() { return _threadMutex_; }
std::map<std::string, bool>& GlobalVariables::getBoolMap() { return _boolMap_; }
std::vector<TChain*>& GlobalVariables::getChainList() { return _chainList_; }
GenericToolbox::ParallelWorker &GlobalVariables::getParallelWorker() {
  return _threadPool_;
}
