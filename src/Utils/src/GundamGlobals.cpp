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
bool GundamGlobals::_debugConfigReading_{false};
bool GundamGlobals::_forceDirectCalculation_{false};
bool GundamGlobals::_lightOutputMode_{false};
std::mutex GundamGlobals::_threadMutex_;
