//
// Created by Nadrino on 11/02/2021.
//

#include "GundamGlobals.h"

// statics
int GundamGlobals::_nbCpuThreads_{1};
bool GundamGlobals::_useCacheManager_{false};
bool GundamGlobals::_isDebug_{false};
bool GundamGlobals::_forceCpuCalculation_{false};
bool GundamGlobals::_lightOutputModeEnabled_{false};
std::mutex GundamGlobals::_threadMutex_;
