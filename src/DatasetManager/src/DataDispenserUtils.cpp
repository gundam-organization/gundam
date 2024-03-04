//
// Created by Adrien Blanchet on 29/09/2023.
//

#include "DataDispenserUtils.h"

#include "GenericToolbox.Map.h"
#include "Logger.h"

#include "sstream"

LoggerInit([]{
  Logger::setUserHeaderStr("[DataDispenserUtils]");
});


std::string DataDispenserParameters::getSummary() const{
  std::stringstream ss;
  ss << GET_VAR_NAME_VALUE(useMcContainer);
  ss << std::endl << GET_VAR_NAME_VALUE(name);
  ss << std::endl << GET_VAR_NAME_VALUE(treePath);
  ss << std::endl << GET_VAR_NAME_VALUE(nominalWeightFormulaStr);
  ss << std::endl << GET_VAR_NAME_VALUE(selectionCutFormulaStr);
  ss << std::endl << "activeLeafNameList = " << GenericToolbox::toString(activeLeafNameList, true);
  ss << std::endl << "filePathList = " << GenericToolbox::toString(filePathList, true);
  ss << std::endl << "variableDict = " << GenericToolbox::toString(variableDict, true);
  ss << std::endl << "additionalVarsStorage = " << GenericToolbox::toString(additionalVarsStorage, true);
  ss << std::endl << GET_VAR_NAME_VALUE(iThrow);
  return ss.str();
}


void DataDispenserCache::clear(){
  propagatorPtr = nullptr;

  samplesToFillList.clear();
  sampleNbOfEvents.clear();
  eventIsInSamplesList.clear();

  sampleIndexOffsetList.clear();
  sampleEventListPtrToFill.clear();

  varsRequestedForIndexing.clear();
  varsRequestedForStorage.clear();
  varToLeafDict.clear();

  varsToOverrideList.clear();

  eventVarTransformList.clear();
}
void DataDispenserCache::addVarRequestedForIndexing(const std::string& varName_) {
  LogThrowIf(varName_.empty(), "no var name provided.");
  GenericToolbox::addIfNotInVector(varName_, this->varsRequestedForIndexing);
}
void DataDispenserCache::addVarRequestedForStorage(const std::string& varName_){
  LogThrowIf(varName_.empty(), "no var name provided.");
  GenericToolbox::addIfNotInVector(varName_, this->varsRequestedForStorage);
  this->addVarRequestedForIndexing(varName_);
}


