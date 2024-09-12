//
// Created by Adrien Blanchet on 29/09/2023.
//

#ifndef GUNDAM_DATA_DISPENSER_UTILS_H
#define GUNDAM_DATA_DISPENSER_UTILS_H

#include "Propagator.h"
#include "EventVarTransformLib.h"

#include "GenericToolbox.Wrappers.h"

#include "nlohmann/json.hpp"

#include "string"
#include "map"


struct DataDispenserParameters{

  // should be load dials and request the associate variables?
  bool useReweightEngine{false};

  std::string name{};
  std::string treePath{};
  std::string dialIndexFormula{};
  std::string nominalWeightFormulaStr{};
  std::string selectionCutFormulaStr{};
  std::vector<std::string> activeLeafNameList{};
  std::vector<std::string> filePathList{};
  std::map<std::string, std::string> variableDict{};
  std::vector<std::string> additionalVarsStorage{};
  std::vector<std::string> dummyVariablesList;
  size_t debugNbMaxEventsToLoad{0};

  JsonType fromHistContent{};
  JsonType overridePropagatorConfig{};

  [[nodiscard]] std::string getSummary() const;
};

struct DataDispenserCache{
  Propagator* propagatorPtr{nullptr};

  size_t totalNbEvents{0};

  std::vector<Sample*> samplesToFillList{};
  std::vector<size_t> sampleNbOfEvents;
  std::vector<std::vector<bool>> eventIsInSamplesList{};
  std::vector<size_t> sampleIndexOffsetList;
  std::vector< std::vector<Event>* > sampleEventListPtrToFill;
  std::vector<DialCollection*> dialCollectionsRefList{};

  std::vector<std::string> varsRequestedForIndexing{};
  std::vector<std::string> varsRequestedForStorage{};
  std::map<std::string, std::pair<std::string, bool>> varToLeafDict; // varToLeafDict[EVENT_VAR_NAME] = {LEAF_NAME, IS_DUMMY}

  std::vector<std::string> varsToOverrideList; // stores the leaves names to override in the right order

  // Variable transformations
  std::vector<EventVarTransformLib> eventVarTransformList;

  struct ThreadSelectionResult{
    std::vector<size_t> sampleNbOfEvents;
    std::vector<std::vector<bool>> eventIsInSamplesList;
  };
  std::vector<ThreadSelectionResult> threadSelectionResults;

  void clear();
  void addVarRequestedForIndexing(const std::string& varName_);
  void addVarRequestedForStorage(const std::string& varName_);

};



#endif //GUNDAM_DATA_DISPENSER_UTILS_H
