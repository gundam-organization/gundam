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

  struct Histogram{
    struct Bin{
      int index{-1};
      double content{0};
      double error{0};
      const DataBin* dataBinPtr{nullptr};
      std::vector<PhysicsEvent*> eventPtrList{};
    };
    std::vector<Bin> binList{};
    int nBins{0};
  };

public:
  SampleElement() = default;

  // setters
  void setName(const std::string& name_){ _name_ = name_; }

  // const-getters
  [[nodiscard]] const std::string& getName() const{ return _name_; }
  [[nodiscard]] const std::vector<PhysicsEvent> &getEventList() const{ return _eventList_; }
  [[nodiscard]] const Histogram &getHistogram() const{ return _histogram_; }

  // mutable-getters
  std::vector<PhysicsEvent> &getEventList(){ return _eventList_; }

  // core
  void buildHistogram(const DataBinSet& binning_);
  void reserveEventMemory(size_t dataSetIndex_, size_t nEvents, const PhysicsEvent &eventBuffer_);
  void shrinkEventList(size_t newTotalSize_);
  void updateBinEventList(int iThread_ = -1);
  void refillHistogram(int iThread_ = -1);

  // event by event poisson throw -> takes into account the finite amount of stat in MC
  void throwEventMcError();

  // generate a toy experiment -> hist content as the asimov -> throw poisson for each bin
  void throwStatError(bool useGaussThrow_ = false);

  [[nodiscard]] double getSumWeights() const;
  [[nodiscard]] size_t getNbBinnedEvents() const;
  [[nodiscard]] std::shared_ptr<TH1D> generateRootHistogram() const; // for the plot generator or for TFile save

  // debug
  [[nodiscard]] std::string getSummary() const;
  friend std::ostream& operator <<( std::ostream& o, const SampleElement& this_ );

private:
  std::string _name_{};
  Histogram _histogram_{};
  std::vector<PhysicsEvent> _eventList_{};
  std::vector<DatasetProperties> _loadedDatasetList_{};

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
