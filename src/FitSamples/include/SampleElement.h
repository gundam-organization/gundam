//
// Created by Adrien BLANCHET on 30/07/2021.
//

#ifndef GUNDAM_SAMPLEELEMENT_H
#define GUNDAM_SAMPLEELEMENT_H

#include "DataBinSet.h"
#include "PhysicsEvent.h"

#include "TH1D.h"

#include "vector"
#include "memory"
#include "string"


class SampleElement{

public:

  SampleElement();
  virtual ~SampleElement();

  std::string name;

  // Events
  std::vector<PhysicsEvent> eventList;

  // Datasets
  std::vector<size_t> dataSetIndexList;
  std::vector<size_t> eventOffSetList;
  std::vector<size_t> eventNbList;

  // Histograms
  DataBinSet binning;
  std::shared_ptr<TH1D> histogram{nullptr};
  std::shared_ptr<TH1D> histogramNominal{nullptr};
  std::vector<std::vector<PhysicsEvent*>> perBinEventPtrList;
  double histScale{1};
  bool isLocked{false};

  // Methods
  void reserveEventMemory(size_t dataSetIndex_, size_t nEvents, const PhysicsEvent &eventBuffer_);
  void shrinkEventList(size_t newTotalSize_);
  void updateEventBinIndexes(int iThread_ = -1);
  void updateBinEventList(int iThread_ = -1);
  void refillHistogram(int iThread_ = -1);
  void rescaleHistogram();
  void saveAsHistogramNominal();

  void throwStatError();

  double getSumWeights() const;
  size_t getNbBinnedEvents() const;

  // debug
  void print() const;

  bool debugTrigger{false};

#ifdef GUNDAM_USING_CACHE_MANAGER
public:
  void setCacheManagerIndex(int i) {_CacheManagerIndex_ = i;}
  int  getCacheManagerIndex() {return _CacheManagerIndex_;}
  void setCacheManagerValuePointer(const double* v) {_CacheManagerValue_ = v;}
  void setCacheManagerValidPointer(const bool* v) {_CacheManagerValid_ = v;}
  void setCacheManagerUpdatePointer(void (*p)()) {_CacheManagerUpdate_ = p;}
private:
  // An "opaque" index into the cache that is used to simplify bookkeeping.
  int _CacheManagerIndex_{-1};
  // A pointer to the cached result.
  const double* _CacheManagerValue_{nullptr};
  // A pointer to the cache validity flag.
  const bool* _CacheManagerValid_{nullptr};
  // A pointer to a callback to force the cache to be updated.
  void (*_CacheManagerUpdate_)(){nullptr};
#endif

};


#endif //GUNDAM_SAMPLEELEMENT_H
