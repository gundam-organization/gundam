//
// Created by Nadrino on 11/02/2021.
//

#ifndef GUNDAM_GUNDAM_GLOBALS_H
#define GUNDAM_GUNDAM_GLOBALS_H

#include <mutex>


class GundamGlobals{

public:
  // setters
  static void setIsCacheManagerEnabled( bool enable){ _isCacheManagerEnabled_ = enable; }
  static void setIsDebug( bool enable){ _debugConfigReading_ = enable; }
  static void setNumberOfThreads(int nbCpuThreads_){ _nbCpuThreads_ = nbCpuThreads_; }
  static void setForceDirectCalculation(bool enable){ _forceDirectCalculation_ = enable; }
  static void setLightOutputMode(bool enable_){ _lightOutputMode_ = enable_; }
  static void setDisableDialCache(bool disableDialCache_){ _disableDialCache_ = disableDialCache_; }

  // getters
  static bool isDebug(){ return _debugConfigReading_; }
  static bool isCacheManagerEnabled(){ return _isCacheManagerEnabled_; }
  static bool getForceDirectCalculation(){ return _forceDirectCalculation_; }
  static bool isDisableDialCache(){ return _disableDialCache_; }
  static bool isLightOutputMode(){ return _lightOutputMode_; }
  static int getNbCpuThreads(){ return _nbCpuThreads_; }
  static std::mutex& getThreadMutex(){ return _threadMutex_; }

private:
  static int _nbCpuThreads_;
  static bool _disableDialCache_;
  static bool _isCacheManagerEnabled_;
  static bool _debugConfigReading_;
  static bool _forceDirectCalculation_;
  static bool _lightOutputMode_;
  static std::mutex _threadMutex_;

};

#endif // GUNDAM_GUNDAM_GLOBALS_H
