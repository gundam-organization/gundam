//
// Created by Nadrino on 11/02/2021.
//

#ifndef GUNDAM_GLOBALVARIABLES_H
#define GUNDAM_GLOBALVARIABLES_H

#include "GenericToolbox.ParallelWorker.h"

#include <TTree.h>
#include <TChain.h>
#include <TRandom3.h>

#include <map>
#include <mutex>


class GlobalVariables{

public:

  // Setters
  static void setNbThreads(int nbThreads_);
  static void setPrngSeed(ULong_t seed_);
  static void setEnableEventWeightCache(bool enable = true);

  // Getters
  static bool isEnableDevMode();
  static const int& getNbThreads();
  static std::mutex& getThreadMutex();
  static std::map<std::string, bool>& getBoolMap();
  static std::vector<TChain*>& getChainList();
  static GenericToolbox::ParallelWorker &getParallelWorker();
  static TRandom3& getPrng();
  static bool getEnableEventWeightCache();

private:

  static bool _enableDevMode_;
  static int _nbThreads_;
  static std::mutex _threadMutex_;
  static std::map<std::string, bool> _boolMap_;
  static std::vector<TChain*> _chainList_;
  static GenericToolbox::ParallelWorker _threadPool_;
  static TRandom3 _prng_;
  static bool _enableEventWeightCache_;

};

#endif // XSLLHFITTER_GLOBALVARIABLES_H
