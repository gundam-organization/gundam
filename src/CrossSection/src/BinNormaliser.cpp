//
// Created by Nadrino on 26/06/2025.
//

#include "BinNormaliser.h"


void BinNormaliser::configureImpl(){
  LogScopeIndent;
  _config_.defineFields({
    {FieldFlag::MANDATORY, "name"},
    {"isEnabled"},
    {"meanValue"},
    {"stdDev"},
    {"disabledBinDim"},
    {"parSetNormName"},
  });
  _config_.checkConfiguration();

  name = _config_.fetchValue<std::string>("name");

  if( not _config_.fetchValue("isEnabled", bool(true)) ){
    LogWarning << "Skipping disabled re-normalization config \"" << name << "\"" << std::endl;
    return;
  }

  LogInfo << "Re-normalization config \"" << name << "\": ";

  if     ( _config_.hasField( "meanValue" ) ){
    normParameter.min  = _config_.fetchValue<double>("meanValue");
    normParameter.max = _config_.fetchValue("stdDev", double(0.));
    LogInfo << "mean ± sigma = " << normParameter.min << " ± " << normParameter.max;
  }
  else if( _config_.hasField("disabledBinDim" ) ){
    disabledBinDim = _config_.fetchValue<std::string>("disabledBinDim");
    LogInfo << "disabledBinDim = " << disabledBinDim;
  }
  else if( _config_.hasField("parSetNormName" ) ){
    parSetNormaliserName = _config_.fetchValue<std::string>("parSetNormName");
    LogInfo << "parSetNormName = " << parSetNormaliserName;
  }
  else{
    LogInfo << std::endl;
    LogThrow("Unrecognized config.");
  }

  LogInfo << std::endl;
}
void BinNormaliser::initializeImpl(){

}

