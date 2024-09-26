//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_SAMPLE_H
#define GUNDAM_SAMPLE_H


#include "Event.h"
#include "DataBinSet.h"

#include <TH1D.h>
#include <TTreeFormula.h>

#include <vector>
#include <string>
#include <memory>


class Sample : public JsonBaseClass {

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
      std::vector<Event*> eventPtrList{};
    };
    std::vector<Bin> binList{};
    int nBins{0};
  };

protected:
  // called through JsonBaseClass::readConfig() and JsonBaseClass::initialize()
  void readConfigImpl() override;

public:
  // SETTERS
  void setIndex(int index){ _index_ = index; }
  void setLlhStatBuffer(double llhStatBuffer_) { _llhStatBuffer_ = llhStatBuffer_; }
  void setName(const std::string &name){ _name_ = name; }
  void setBinningFilePath(const JsonType &binningFilePath_){ _binningConfig_ = binningFilePath_; }
  void setSelectionCutStr(const std::string &selectionCutStr_){ _selectionCutStr_ = selectionCutStr_; }
  void setEnabledDatasetList(const std::vector<std::string>& enabledDatasetList_){ _enabledDatasetList_ = enabledDatasetList_; }

  // const getters
  [[nodiscard]] bool isEnabled() const{ return _isEnabled_; }
  [[nodiscard]] int getIndex() const{ return _index_; }
  [[nodiscard]] double getLlhStatBuffer() const { return _llhStatBuffer_; }
  [[nodiscard]] const std::string &getName() const{ return _name_; }
  [[nodiscard]] const std::string &getSelectionCutsStr() const{ return _selectionCutStr_; }
  [[nodiscard]] const JsonType &getBinningFilePath() const{ return _binningConfig_; }
  [[nodiscard]] const DataBinSet &getBinning() const{ return _binning_; }
  [[nodiscard]] const Histogram &getHistogram() const{ return _histogram_; }
  [[nodiscard]] const std::vector<Event> &getEventList() const{ return _eventList_; }

  // getters
  DataBinSet &getBinning() { return _binning_; }
  std::vector<Event> &getEventList(){ return _eventList_; }

  // misc
  void writeEventRates(const GenericToolbox::TFilePath& saveDir_) const;
  bool isDatasetValid(const std::string& datasetName_);

  // core
  void buildHistogram(const DataBinSet& binning_);
  void reserveEventMemory(size_t dataSetIndex_, size_t nEvents, const Event &eventBuffer_);
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

  // printouts
  void printConfiguration() const;
  [[nodiscard]] std::string getSummary() const;
  friend std::ostream& operator <<( std::ostream& o, const Sample& this_ );

private:
  // configuration
  bool _isEnabled_{true};
  int _index_{-1};
  std::string _name_;
  std::string _selectionCutStr_;
  JsonType _binningConfig_;
  std::vector<std::string> _enabledDatasetList_;

  // Internals
  double _llhStatBuffer_{std::nan("unset")}; // set by SampleSet which hold the joinProbability obj
  DataBinSet _binning_;
  std::vector<size_t> _dataSetIndexList_;

  Histogram _histogram_{};
  std::vector<Event> _eventList_{};
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


#endif //GUNDAM_SAMPLE_H
