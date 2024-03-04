//
// Created by Nadrino on 04/03/2024.
//

#ifndef GUNDAM_DATASET_MANAGER_H
#define GUNDAM_DATASET_MANAGER_H

#include "DatasetDefinition.h"
#include "JsonBaseClass.h"


class DataSetManager : public JsonBaseClass {

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

public:
  DataSetManager() = default;

  // const-getters
  [[nodiscard]] const EventTreeWriter& getTreeWriter() const{ return _treeWriter_; }
  [[nodiscard]] const std::vector<DatasetDefinition>& getDataSetList() const{ return _dataSetList_; }

  // mutable-getters
  EventTreeWriter& getTreeWriter(){ return _treeWriter_; }
  std::vector<DatasetDefinition>& getDataSetList(){ return _dataSetList_; }

private:
  // internals
  EventTreeWriter _treeWriter_{};
  std::vector<DatasetDefinition> _dataSetList_{};

};


#endif //GUNDAM_DATASET_MANAGER_H
