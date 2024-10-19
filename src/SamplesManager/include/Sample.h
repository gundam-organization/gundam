//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_SAMPLE_H
#define GUNDAM_SAMPLE_H


#include "Event.h"
#include "Histogram.h"
#include "BinSet.h"
#include "GundamGlobals.h"

#include "GenericToolbox.Root.h"
#include "GenericToolbox.Loops.h"

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

protected:
  // called through JsonBaseClass::configure() and JsonBaseClass::initialize()
  void configureImpl() override;

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
  [[nodiscard]] bool isEventMcThrowDisabled() const{ return _disableEventMcThrow_; }
  [[nodiscard]] int getIndex() const{ return _index_; }
  [[nodiscard]] double getLlhStatBuffer() const { return _llhStatBuffer_; }
  [[nodiscard]] const std::string &getName() const{ return _name_; }
  [[nodiscard]] const std::string &getSelectionCutsStr() const{ return _selectionCutStr_; }
  [[nodiscard]] const JsonType &getBinningFilePath() const{ return _binningConfig_; }
  [[nodiscard]] const Histogram &getHistogram() const{ return _histogram_; }
  [[nodiscard]] const std::vector<Event> &getEventList() const{ return _eventList_; }

  // getters
  Histogram &getHistogram(){ return _histogram_; }
  std::vector<Event> &getEventList(){ return _eventList_; }

  // const core
  void writeEventRates(const GenericToolbox::TFilePath& saveDir_) const;
  bool isDatasetValid(const std::string& datasetName_) const;
  [[nodiscard]] double getSumWeights() const;
  [[nodiscard]] size_t getNbBinnedEvents() const;

  // core
  void reserveEventMemory(size_t dataSetIndex_, size_t nEvents, const Event &eventBuffer_);
  void shrinkEventList(size_t newTotalSize_);

  // printouts
  void printConfiguration() const;
  [[nodiscard]] std::string getSummary() const;
  friend std::ostream& operator <<( std::ostream& o, const Sample& this_ );

  // multi-thread
  void indexEventInHistogramBin( int iThread_ = -1);

private:
  // configuration
  bool _isEnabled_{true};
  bool _disableEventMcThrow_{false};
  int _index_{-1};
  std::string _name_;
  std::string _selectionCutStr_;
  JsonType _binningConfig_;
  std::vector<std::string> _enabledDatasetList_;

  // Internals
  double _llhStatBuffer_{std::nan("unset")}; // set by SampleSet which hold the joinProbability obj
  std::vector<size_t> _dataSetIndexList_;

  Histogram _histogram_{};
  std::vector<Event> _eventList_{};
  std::vector<DatasetProperties> _loadedDatasetList_{};

};


#endif //GUNDAM_SAMPLE_H
