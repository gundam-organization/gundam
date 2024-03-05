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

  // setters
  void setIsLocked(bool isLocked_){ _isLocked_ = isLocked_; }
  void setName(const std::string& name_){ _name_ = name_; }
  void setBinning(const DataBinSet& binning_){ _binning_ = binning_; }

  // const-getters
  [[nodiscard]] bool isLocked() const{ return _isLocked_; }
  [[nodiscard]] const std::string& getName() const{ return _name_; }
  [[nodiscard]] const DataBinSet& getBinning() const{ return _binning_; }
  [[nodiscard]] const TH1D* getHistogram() const{ return _histogram_.get(); }
  [[nodiscard]] const TH1D* getHistogramNominal() const{ return _histogramNominal_.get(); }
  [[nodiscard]] const std::vector<PhysicsEvent> &getEventList() const{ return _eventList_; }
  [[nodiscard]] const std::vector<std::vector<PhysicsEvent *>> &getPerBinEventPtrList() const{ return _perBinEventPtrList_; }
  [[nodiscard]] const std::vector<DatasetProperties> &getLoadedDatasetList() const{ return _loadedDatasetList_; }

  // mutable-getters
  TH1D* getHistogram(){ return _histogram_.get(); }
  TH1D* getHistogramNominal(){ return _histogramNominal_.get(); }
  std::vector<PhysicsEvent> &getEventList(){ return _eventList_; }
  std::vector<std::vector<PhysicsEvent *>> &getPerBinEventPtrList(){ return _perBinEventPtrList_; }
  std::shared_ptr<TH1D>& getHistogramSharedPtr() { return _histogram_; }

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
  bool _isLocked_{false};
  std::string _name_{};
  DataBinSet _binning_{};
  std::shared_ptr<TH1D> _histogram_{nullptr};
  std::shared_ptr<TH1D> _histogramNominal_{nullptr};
  std::vector<std::vector<PhysicsEvent*>> _perBinEventPtrList_;
  std::vector<PhysicsEvent> _eventList_;
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
