//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_DATASET_DEFINITION_H
#define GUNDAM_DATASET_DEFINITION_H

#include "DataDispenser.h"
#include "ParameterSet.h"
#include <SampleSet.h>
#include "PlotGenerator.h"
#include "Propagator.h"

#include <TChain.h>

#include <vector>
#include <string>


class DatasetDefinition : public JsonBaseClass {

protected:
  void configureImpl() override;
  void initializeImpl() override;

public:
  explicit DatasetDefinition( const JsonType& config_, int datasetIndex_): _dataSetIndex_(datasetIndex_) { this->configure(config_); }

  void setDataSetIndex(int dataSetIndex){ _dataSetIndex_ = dataSetIndex; }
  void setSelectedDataEntry(const std::string& selectedDataEntry_){ _selectedDataEntry_ = selectedDataEntry_; }

  // const-getters
  [[nodiscard]] bool isEnabled() const{ return _isEnabled_; }
  [[nodiscard]] bool isSortLoadedEvents() const{ return _sortLoadedEvents_; }
  [[nodiscard]] bool isShowSelectedEventCount() const{ return _showSelectedEventCount_; }
  [[nodiscard]] bool isDevSingleThreadEventSelection() const{ return _devSingleThreadEventSelection_; }
  [[nodiscard]] bool isDevSingleThreadEventLoaderAndIndexer() const{ return _devSingleThreadEventLoaderAndIndexer_; }
  [[nodiscard]] int getDataSetIndex() const{ return _dataSetIndex_; }
  [[nodiscard]] const std::string &getName() const{ return _name_; }
  [[nodiscard]] const std::string &getSelectedDataEntry() const{ return _selectedDataEntry_; }
  [[nodiscard]] const std::string &getSelectedToyEntry() const{ return _selectedToyEntry_; }
  [[nodiscard]] const std::map<std::string, DataDispenser> &getDataDispenserDict() const { return _dataDispenserDict_; }

  DataDispenser &getModelDispenser(){ return _modelDispenser_; }
  std::map<std::string, DataDispenser> &getDataDispenserDict(){ return _dataDispenserDict_; }

  void updateDispenserOwnership();

  void setNbMaxEventToLoad( size_t nbMaxEventToLoad_ ){
    _modelDispenser_.getParameters().debugNbMaxEventsToLoad = nbMaxEventToLoad_;
    for( auto& dd : _dataDispenserDict_ ){
      dd.second.getParameters().debugNbMaxEventsToLoad = nbMaxEventToLoad_;
    }
  }
  void setFractionOfEntriesToLoad(double fraction_){
    _modelDispenser_.getParameters().fractionOfEntries = fraction_;
    for( auto& dd : _dataDispenserDict_ ){
      dd.second.getParameters().fractionOfEntries = fraction_;
    }
  }

private:
  // config
  bool _isEnabled_{true};
  bool _showSelectedEventCount_{true};
  int _dataSetIndex_{-1};
  std::string _name_{};
  std::string _selectedDataEntry_{"Asimov"};
  std::string _selectedToyEntry_{"Asimov"};

  bool _sortLoadedEvents_{true}; // needed for reproducibility of toys in stat throw
  bool _devSingleThreadEventLoaderAndIndexer_{false};
  bool _devSingleThreadEventSelection_{false};

  // internals
  DataDispenser _modelDispenser_{this};
  std::map<std::string, DataDispenser> _dataDispenserDict_;

};


#endif //GUNDAM_DATASET_DEFINITION_H
