//
// Created by Nadrino on 11/02/2021.
//

#ifndef GUNDAM_GUNDAM_GLOBALS_H
#define GUNDAM_GUNDAM_GLOBALS_H

#include <TTree.h>
#include <TChain.h>
#include <TRandom3.h>

#include <map>
#include <mutex>
#include <memory>

#define ENUM_NAME VerboseLevel
#define ENUM_FIELDS \
  ENUM_FIELD(NORMAL_MODE, 0) \
  ENUM_FIELD(MORE_PRINTOUT) \
  ENUM_FIELD(DEBUG_TRACE) \
  ENUM_FIELD(INLOOP_TRACE) \
  ENUM_FIELD(DEV_TRACE)
#include "GenericToolbox.MakeEnum.h"

class GundamGlobals{

public:

  // Setters
  static void setEnableCacheManager(bool enable = true){ _enableCacheManager_ = enable; }
  static void setNumberOfThreads(int threads=1){ _gundamThreads_ = threads; }
  static void setForceDirectCalculation(bool enable=false){ _forceDirectCalculation_ = enable; }
  static void setLightOutputMode(bool enable_){ _lightOutputMode_ = enable_; }
  static void setDisableDialCache(bool disableDialCache_){ _disableDialCache_ = disableDialCache_; }
  static void setVerboseLevel(VerboseLevel verboseLevel_);

  // Getters
  static int getNumberOfThreads(){ return _gundamThreads_; }
  static bool getEnableCacheManager(){ return _enableCacheManager_; }
  static bool getForceDirectCalculation(){ return _forceDirectCalculation_; }
  static bool isDisableDialCache(){ return _disableDialCache_; }
  static bool isLightOutputMode(){ return _lightOutputMode_; }
  static VerboseLevel::EnumType getVerboseLevel(){ return _verboseLevel_.value; }
  static std::mutex& getThreadMutex(){ return _threadMutex_; }

private:

  static int _gundamThreads_;
  static bool _disableDialCache_;
  static bool _enableCacheManager_;
  static bool _forceDirectCalculation_;
  static bool _lightOutputMode_;
  static std::mutex _threadMutex_;
  static VerboseLevel _verboseLevel_;

};

#endif // GUNDAM_GUNDAM_GLOBALS_H
