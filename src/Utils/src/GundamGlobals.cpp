//
// Created by Nadrino on 11/02/2021.
//

#include "GundamGlobals.h"
#include "Logger.h"

#include "TRandom3.h"

#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[GlobalVariables]"); });
#endif

// statics
int GundamGlobals::_gundamThreads_{1};
bool GundamGlobals::_disableDialCache_{false};
bool GundamGlobals::_enableCacheManager_{false};
bool GundamGlobals::_forceDirectCalculation_{false};
bool GundamGlobals::_lightOutputMode_{false};
std::mutex GundamGlobals::_threadMutex_;
VerboseLevel GundamGlobals::_verboseLevel_{VerboseLevel::NORMAL_MODE};

// setters
void GundamGlobals::setVerboseLevel(VerboseLevel verboseLevel_){
  _verboseLevel_ = verboseLevel_;
  LogWarning << "Verbose level set to: " << _verboseLevel_.toString() << std::endl;
}
