//
// Created by Nadrino on 11/02/2021.
//

#include "GundamGlobals.h"
#include "Logger.h"

#include "TRandom3.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[GlobalVariables]");
});

// statics
int GundamGlobals::_gundamThreads_{1};
bool GundamGlobals::_disableDialCache_{false};
bool GundamGlobals::_enableCacheManager_{false};
bool GundamGlobals::_forceDirectCalculation_{false};
bool GundamGlobals::_lightOutputMode_{false};
std::mutex GundamGlobals::_threadMutex_;
VerboseLevel GundamGlobals::_verboseLevel_{NORMAL_MODE};
GenericToolbox::ParallelWorker GundamGlobals::_threadPool_;

// setters
void GundamGlobals::setVerboseLevel(VerboseLevel verboseLevel_){
  _verboseLevel_ = verboseLevel_;
  LogWarning << "Verbose level set to: " << VerboseLevelEnumNamespace::toString(_verboseLevel_) << std::endl;
}
void GundamGlobals::setVerboseLevel(int verboseLevel_){ GundamGlobals::setVerboseLevel(static_cast<VerboseLevel>(verboseLevel_)); }
