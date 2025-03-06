//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_SAMPLE_H
#define GUNDAM_SAMPLE_H


#include "Event.h"
#include "Histogram.h"
#include "BinSet.h"
#include "GundamGlobals.h"

#include "GenericToolbox.Root.h" // TFile
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
  // setters
  void setIndex(int index){ _index_ = index; }
  void setName(const std::string &name){ _name_ = name; }
  void setBinningFilePath(const JsonType &binningFilePath_){ _binningConfig_ = binningFilePath_; }

  // const getters
  [[nodiscard]] auto isEnabled() const{ return _isEnabled_; }
  [[nodiscard]] auto isEventMcThrowDisabled() const{ return _disableEventMcThrow_; }
  [[nodiscard]] auto getIndex() const{ return _index_; }
  [[nodiscard]] auto& getName() const{ return _name_; }
  [[nodiscard]] auto& getSelectionCutsStr() const{ return _selectionCutStr_; }
  [[nodiscard]] auto& getBinningFilePath() const{ return _binningConfig_; }
  [[nodiscard]] auto& getHistogram() const{ return _histogram_; }
  [[nodiscard]] auto& getEventList() const{ return _eventList_; }

  // mutable getters
  auto& getHistogram(){ return _histogram_; }
  auto& getEventList(){ return _eventList_; }

  // const core
  void writeEventRates(const GenericToolbox::TFilePath& saveDir_) const;
  [[nodiscard]] bool isDatasetValid(const std::string& datasetName_) const;
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
  std::vector<size_t> _dataSetIndexList_;

  Histogram _histogram_{};
  std::vector<Event> _eventList_{};
  std::vector<DatasetProperties> _loadedDatasetList_{};

};


#endif //GUNDAM_SAMPLE_H
