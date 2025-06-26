//
// Created by Nadrino on 25/06/2025.
//

#include "ParameterSetRenorm.h"

void ParameterSetRenorm::configureImpl(){
  _config_.clearFields();
  _config_.defineFields({
    {FieldFlag::MANDATORY, "name"},
    {FieldFlag::MANDATORY, "filePath"},
    {FieldFlag::MANDATORY, "histogramPath"},
    {FieldFlag::MANDATORY, "axisVariable"},
    {"parSelections"},
  });
  _config_.checkConfiguration();


}
void ParameterSetRenorm::initializeImpl(){

}
