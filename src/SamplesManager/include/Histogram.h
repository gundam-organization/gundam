//
// Created by Nadrino on 19/10/2024.
//

#ifndef GUNDAM_HISTOGRAM_H
#define GUNDAM_HISTOGRAM_H

#include "Event.h"
#include "Bin.h"
#include "ConfigUtils.h"

#include "GenericToolbox.Loops.h"


class Histogram{

public:
  // structs
  struct BinContent{
    // keeps the contents close together in memory
    double sumWeights{0};
    double sqrtSumSqWeights{0};
  };
  struct BinContext{
    Bin bin{};
    std::vector<Event*> eventPtrList{};
  };

  // const getters
  [[nodiscard]] int getNbBins() const { return nBins; }
  [[nodiscard]] const std::vector<BinContent>& getBinContentList() const { return binContentList; }
  [[nodiscard]] const std::vector<BinContext>& getBinContextList() const { return binContextList; }

  // mutable getters
  std::vector<BinContent>& getBinContentList(){ return binContentList; }
  std::vector<BinContext>& getBinContextList(){ return binContextList; }

  // core
  void build(const JsonType& binningConfig_);
  void throwEventMcError();
  void throwStatError(bool useGaussThrow_ = false);

  // multi-thread
  void updateBinEventList(std::vector<Event>& eventList_, int iThread_ = -1);
  void refillHistogram(int iThread_ = -1);

  // utils
  auto loop(){ return GenericToolbox::Zip(binContentList, binContextList); }
  auto loop(size_t start_, size_t end_){ return GenericToolbox::ZipPartial(start_, end_, binContentList, binContextList); }
  [[nodiscard]] auto loop() const{ return GenericToolbox::Zip(binContentList, binContextList); }
  [[nodiscard]] auto loop(size_t start_, size_t end_) const { return GenericToolbox::ZipPartial(start_, end_, binContentList, binContextList); }

private:
  int nBins{0};
  std::vector<BinContent> binContentList{};
  std::vector<BinContext> binContextList{};

#ifdef GUNDAM_USING_CACHE_MANAGER
public:
  [[nodiscard]] bool isCacheManagerEnabled() const { return _cacheManagerIndex_ >= 0 and _cacheManagerValidFlagPtr_ and (*_cacheManagerValidFlagPtr_); };

  void setCacheManagerIndex(int i) { _cacheManagerIndex_ = i;}
  void setCacheManagerValuePointer(const double* v) { _cacheManagerSumWeightsArray_ = v;}
  void setCacheManagerValue2Pointer(const double* v) { _cacheManagerSumSqWeightsArray_ = v;}
  void setCacheManagerValidPointer(const bool* v) { _cacheManagerValidFlagPtr_ = v;}
  void setCacheManagerUpdatePointer(void (*p)()) { _cacheManagerUpdateFctPtr_ = p;}

  [[nodiscard]] void (*getCacheManagerUpdateFctPtr() const)() { return _cacheManagerUpdateFctPtr_; }
  [[nodiscard]] int getCacheManagerIndex() const {return _cacheManagerIndex_;}
  [[nodiscard]] const bool* getCacheManagerValidFlagPtr() const { return _cacheManagerValidFlagPtr_; }
  [[nodiscard]] const double* getCacheManagerSumWeightsArray() const { return _cacheManagerSumWeightsArray_; }
  [[nodiscard]] const double* getCacheManagerSumSqWeightsArray() const { return _cacheManagerSumSqWeightsArray_; }

private:
  // An "opaque" index into the cache that is used to simplify bookkeeping.
  int _cacheManagerIndex_{-1};
  // A pointer to the cached result.
  const double* _cacheManagerSumWeightsArray_{nullptr};
  // A pointer to the cached result.
  const double* _cacheManagerSumSqWeightsArray_{nullptr};
  // A pointer to the cache validity flag.
  const bool* _cacheManagerValidFlagPtr_{nullptr};
  // A pointer to a callback to force the cache to be updated.
  void (*_cacheManagerUpdateFctPtr_)(){nullptr};
#endif

};


#endif //GUNDAM_HISTOGRAM_H
