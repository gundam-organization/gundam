#ifndef GUNDAM_DATASET_MANAGER_H
#define GUNDAM_DATASET_MANAGER_H

#include "DatasetDefinition.h"
#include "DatasetDefinitionTemplate.h"
#include "ConfigUtils.h"

#include <vector>

class DatasetManager : public JsonBaseClass {
protected:
  void configureImpl() override;
  void initializeImpl() override;

public:
  [[nodiscard]] auto& getDatasetList() const { return _datasetList_; }
  auto& getDatasetList(){ return _datasetList_; }

private:
  std::vector<DatasetDefinitionTemplate> _datasetTemplateList_{};
  std::vector<DatasetDefinition> _datasetList_{};
};

#endif // GUNDAM_DATASET_MANAGER_H
