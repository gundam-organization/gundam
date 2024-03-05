//
// Created by Adrien BLANCHET on 30/07/2021.
//

#ifndef GUNDAM_SAMPLE_ELEMENT_H
#define GUNDAM_SAMPLE_ELEMENT_H

#include "DataBinSet.h"
#include "PhysicsEvent.h"

#include "TH1D.h"

#include <vector>
#include <memory>
#include <string>


class SampleElement{

public:
  struct DatasetProperties{
    size_t dataSetIndex{0};
    size_t eventOffSet{0};
    size_t eventNb{0};
  };

public:
  SampleElement() = default;

  // const-getters
  [[nodiscard]] bool isLocked1() const{ return isLocked; }
  [[nodiscard]] const std::string& getName() const{ return name; }
  [[nodiscard]] const DataBinSet& getBinning() const{ return binning; }
  [[nodiscard]] const TH1D* getHistogram() const{ return histogram.get(); }
  [[nodiscard]] const TH1D* getHistogramNominal() const{ return histogramNominal.get(); }
  [[nodiscard]] const std::vector<std::vector<PhysicsEvent *>> &getPerBinEventPtrList() const{ return perBinEventPtrList; }
  [[nodiscard]] const std::vector<PhysicsEvent> &getEventList() const{ return eventList; }
  [[nodiscard]] const std::vector<DatasetProperties> &getLoadedDatasetList() const{ return loadedDatasetList; }

  // mutable-getters
  TH1D* getHistogram(){ return histogram.get(); }
  TH1D* getHistogramNominal(){ return histogramNominal.get(); }

  // Methods
  void reserveEventMemory(size_t dataSetIndex_, size_t nEvents, const PhysicsEvent &eventBuffer_);
  void shrinkEventList(size_t newTotalSize_);
  void updateEventBinIndexes(int iThread_ = -1);
  void updateBinEventList(int iThread_ = -1);
  void refillHistogram(int iThread_ = -1);
  void saveAsHistogramNominal();

  // event by event poisson throw -> takes into account the finite amount of stat in MC
  void throwEventMcError();

  // generate a toy experiment -> hist content as the asimov -> throw poisson for each bin
  void throwStatError(bool useGaussThrow_ = false);

  [[nodiscard]] double getSumWeights() const;
  [[nodiscard]] size_t getNbBinnedEvents() const;

  // debug
  [[nodiscard]] std::string getSummary() const;
  friend std::ostream& operator <<( std::ostream& o, const SampleElement& this_ );

private:
  bool isLocked{false};
  std::string name{};
  DataBinSet binning;
  std::shared_ptr<TH1D> histogram{nullptr};
  std::shared_ptr<TH1D> histogramNominal{nullptr};
  std::vector<std::vector<PhysicsEvent*>> perBinEventPtrList;
  std::vector<PhysicsEvent> eventList;
  std::vector<DatasetProperties> loadedDatasetList{};

#ifdef GUNDAM_USING_CACHE_MANAGER
public:
  void setCacheManagerIndex(int i) {_CacheManagerIndex_ = i;}
  void setCacheManagerValuePointer(const double* v) {_CacheManagerValue_ = v;}
  void setCacheManagerValue2Pointer(const double* v) {_CacheManagerValue2_ = v;}
  void setCacheManagerValidPointer(const bool* v) {_CacheManagerValid_ = v;}
  void setCacheManagerUpdatePointer(void (*p)()) {_CacheManagerUpdate_ = p;}

  [[nodiscard]] int getCacheManagerIndex() const {return _CacheManagerIndex_;}
private:
  // An "opaque" index into the cache that is used to simplify bookkeeping.
  int _CacheManagerIndex_{-1};
  // A pointer to the cached result.
  const double* _CacheManagerValue_{nullptr};
  // A pointer to the cached result.
  const double* _CacheManagerValue2_{nullptr};
  // A pointer to the cache validity flag.
  const bool* _CacheManagerValid_{nullptr};
  // A pointer to a callback to force the cache to be updated.
  void (*_CacheManagerUpdate_)(){nullptr};
#endif

};


#endif //GUNDAM_SAMPLE_ELEMENT_H
