#ifndef GUNDAM_DATASET_DEFINITION_TEMPLATES_H
#define GUNDAM_DATASET_DEFINITION_TEMPLATES_H

#include "ConfigUtils.h"

#include <string>

// Holds a partial dataset configuration. Its fields are validated only after
// the templates and the dataset configuration have been combined.
class DatasetDefinitionTemplates : public JsonBaseClass {
protected:
  void configureImpl() override;
  void initializeImpl() override {}

public:
  [[nodiscard]] auto& getName() const { return _name_; }

private:
  std::string _name_{};
};

#endif // GUNDAM_DATASET_DEFINITION_TEMPLATES_H
