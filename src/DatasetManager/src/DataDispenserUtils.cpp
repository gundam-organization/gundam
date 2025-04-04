//
// Created by Adrien Blanchet on 29/09/2023.
//

#include "DataDispenserUtils.h"

#include "GenericToolbox.Map.h"
#include "Logger.h"

#include "sstream"


std::string DataDispenserParameters::getSummary() const{
  std::stringstream ss;
  ss << GET_VAR_NAME_VALUE(useReweightEngine);
  ss << std::endl << GET_VAR_NAME_VALUE(name);
  ss << std::endl << GET_VAR_NAME_VALUE(globalTreePath);
  ss << std::endl << GET_VAR_NAME_VALUE(nominalWeightFormulaStr);
  ss << std::endl << GET_VAR_NAME_VALUE(selectionCutFormulaStr);
  ss << std::endl << "activeLeafNameList = " << GenericToolbox::toString(activeLeafNameList, true);
  ss << std::endl << "filePathList = " << GenericToolbox::toString(filePathList, true);
  ss << std::endl << "variableDict = " << GenericToolbox::toString(variableDict, true);
  ss << std::endl << "additionalVarsStorage = " << GenericToolbox::toString(additionalVarsStorage, true);
  return ss.str();
}


void DataDispenserCache::clear(){
  propagatorPtr = nullptr;

  samplesToFillList.clear();
  sampleNbOfEvents.clear();
  entrySampleIndexList.clear();

  sampleIndexOffsetList.clear();
  sampleEventListPtrToFill.clear();

  varsRequestedForIndexing.clear();
  varToLeafDict.clear();

  varsToOverrideList.clear();
}
void DataDispenserCache::addVarRequestedForIndexing(const std::string& varName_) {
  LogThrowIf(varName_.empty(), "no var name provided.");
  GenericToolbox::addIfNotInVector(varName_, this->varsRequestedForIndexing);
}


