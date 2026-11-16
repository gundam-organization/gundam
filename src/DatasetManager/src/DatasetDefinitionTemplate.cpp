#include "DatasetDefinitionTemplate.h"

#include "Logger.h"

void DatasetDefinitionTemplate::configureImpl(){
  _config_.clearFields();
  _config_.defineFields({{FieldFlag::MANDATORY, "name"}});
  // Other fields are deliberately unrestricted: this is a partial dataset.
  _name_ = _config_.fetchValue<std::string>("name");
  LogThrowIf(_name_.empty(), "Dataset template name must not be empty.");
}
