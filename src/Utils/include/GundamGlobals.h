//
// Created by Nadrino on 11/02/2021.
//

#ifndef GUNDAM_GUNDAM_GLOBALS_H
#define GUNDAM_GUNDAM_GLOBALS_H

#include <mutex>


class GundamGlobals{

public:
  // setters
  static void setIsDebug( bool enable){ _isDebug_ = enable; }
  static void setLightOutputMode(bool enable_){ _lightOutputModeEnabled_ = enable_; }
  static void setIsCacheManagerEnabled( bool enable){ _useCacheManager_ = enable; }
  static void setIsForceCpuCalculation( bool enable){ _forceCpuCalculation_ = enable; }
  static void setNumberOfThreads(int nbCpuThreads_){ _nbCpuThreads_ = nbCpuThreads_; }

  // getters
  static bool isDebug(){ return _isDebug_; }
  static bool isLightOutputMode(){ return _lightOutputModeEnabled_; }
  static bool isCacheManagerEnabled(){ return _useCacheManager_; }
  static bool isForceCpuCalculation(){ return _forceCpuCalculation_; }
  static int getNbCpuThreads(){ return _nbCpuThreads_; }

private:
  static bool _isDebug_; /* enable debug printouts for the config reading */
  static bool _useCacheManager_; /* enable the use of the cache manager in the propagator (mainly for GPU) */
  static bool _forceCpuCalculation_; /* force using CPU in the cache manager */
  static bool _lightOutputModeEnabled_;
  static int _nbCpuThreads_;

};

#endif // GUNDAM_GUNDAM_GLOBALS_H
