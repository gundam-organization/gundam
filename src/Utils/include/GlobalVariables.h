//
// Created by Nadrino on 11/02/2021.
//

#ifndef GUNDAM_GLOBALVARIABLES_H
#define GUNDAM_GLOBALVARIABLES_H

#include "GenericToolbox.ParallelWorker.h"
#include "GenericToolbox.h"

#include <TTree.h>
#include <TChain.h>
#include <TRandom3.h>

#include <map>
#include <mutex>
#include <memory>

#ifndef GUNDAM_BATCH
#define GUNDAM_SIGMA "σ"
#define GUNDAM_CHI2 "χ²"
#define GUNDAM_DELTA "Δ"
#else
#define GUNDAM_SIGMA "sigma"
#define GUNDAM_CHI2 "chi-squared"
#define GUNDAM_DELTA "delta-"
#endif

#define USE_BREAKDOWN_CACHE

ENUM_EXPANDER(
    VerboseLevel, 0,
    NORMAL_MODE,
    MORE_PRINTOUT,
    DEBUG_TRACE,
    INLOOP_TRACE,
    DEV_TRACE
    );


class GlobalVariables{

public:

  // Setters
  static void setEnableCacheManager(bool enable = true);
  static void setDisableDialCache(bool disableDialCache_);
  static void setVerboseLevel(VerboseLevel verboseLevel_);
  static void setVerboseLevel(int verboseLevel_);
  static void setNbThreads(int nbThreads_);

  // Getters
  static bool isEnableDevMode();
  static bool getEnableCacheManager();
  static bool isDisableDialCache();
  static VerboseLevel getVerboseLevel();
  static const int& getNbThreads();
  static std::mutex& getThreadMutex();
  static std::map<std::string, bool>& getBoolMap();
  static std::vector<TChain*>& getChainList();
  static GenericToolbox::ParallelWorker &getParallelWorker();

private:

  static bool _enableDevMode_;
  static bool _disableDialCache_;
  static bool _enableCacheManager_;
  static int _nbThreads_;
  static std::mutex _threadMutex_;
  static std::map<std::string, bool> _boolMap_;
  static std::vector<TChain*> _chainList_;
  static VerboseLevel _verboseLevel_;
  static GenericToolbox::ParallelWorker _threadPool_;

};

#endif
