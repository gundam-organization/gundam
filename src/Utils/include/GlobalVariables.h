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

#define USE_NEW_DIALS 1
#define USE_ZLIB 0
#define USE_MANUAL_CACHE 0
//#define USE_TSPLINE3_EVAL


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
  static void setVerboseLevel(VerboseLevel verboseLevel_);
  static void setVerboseLevel(int verboseLevel_);
  static void setNbThreads(int nbThreads_);
  static void setEnableCacheManager(bool enable = true);

  // Getters
  static VerboseLevel getVerboseLevel();
  static bool isEnableDevMode();
  static const int& getNbThreads();
  static std::mutex& getThreadMutex();
  static std::map<std::string, bool>& getBoolMap();
  static std::vector<TChain*>& getChainList();
  static GenericToolbox::ParallelWorker &getParallelWorker();
  static bool getEnableCacheManager();

private:

  static bool _enableDevMode_;
  static int _nbThreads_;
  static std::mutex _threadMutex_;
  static std::map<std::string, bool> _boolMap_;
  static std::vector<TChain*> _chainList_;
  static GenericToolbox::ParallelWorker _threadPool_;
  static bool _enableCacheManager_;
  static VerboseLevel _verboseLevel_;

};

#endif
