//
// Created by Nadrino on 04/03/2024.
//

#ifndef GUNDAM_DATASET_MANAGER_H
#define GUNDAM_DATASET_MANAGER_H

#include "DatasetDefinition.h"
#include "EventTreeWriter.h"
#include "Propagator.h"
#include "JsonBaseClass.h"


class DataSetManager : public JsonBaseClass {

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

public:
  DataSetManager() = default;

  // setters
  void setToyParameterInjector(const JsonType& toyParameterInjector_){ _toyParameterInjector_ = toyParameterInjector_; }

  // const-getters
  [[nodiscard]] const Propagator& getPropagator() const{ return _propagator_; }
  [[nodiscard]] const EventTreeWriter& getTreeWriter() const{ return _treeWriter_; }
  [[nodiscard]] const std::vector<DatasetDefinition>& getDataSetList() const{ return _dataSetList_; }

  // mutable-getters
  Propagator& getPropagator(){ return _propagator_; }
  EventTreeWriter& getTreeWriter(){ return _treeWriter_; }
  std::vector<DatasetDefinition>& getDataSetList(){ return _dataSetList_; }

protected:
  void load();
  void loadPropagator(bool isModel_);

private:
  // internals
  bool _reloadModelRequested_{false};

  Propagator _propagator_{};
  EventTreeWriter _treeWriter_{};
  std::vector<DatasetDefinition> _dataSetList_{};

  JsonType _toyParameterInjector_{};

  GenericToolbox::ParallelWorker _threadPool_{};

};


#endif //GUNDAM_DATASET_MANAGER_H
