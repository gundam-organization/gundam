//
// Created by Adrien Blanchet on 29/09/2023.
//

#ifndef GUNDAM_DATA_DISPENSER_UTILS_H
#define GUNDAM_DATA_DISPENSER_UTILS_H

#include "Propagator.h"
#include "EventVarTransformLib.h"

#include "GenericToolbox.Wrappers.h"
#include "GenericToolbox.Root.h"

#include "TChain.h"
#include "TTreeFormula.h"


#include "string"
#include "map"


struct DataDispenserParameters{

  // should be load dials and request the associate variables?
  bool useReweightEngine{false};
  bool isData{false}; // shall fetch slpit vars?
  size_t debugNbMaxEventsToLoad{0};
  double fractionOfEntries{1.};

  std::string name{};
  std::string globalTreePath{};
  std::string dialIndexFormula{};
  std::string nominalWeightFormulaStr{};
  std::string selectionCutFormulaStr{};
  std::string eventVariableAsWeight{};
  std::vector<std::string> activeLeafNameList{};
  std::vector<std::string> filePathList{};
  std::map<std::string, std::string> variableDict{};
  std::vector<std::string> additionalVarsStorage{};
  std::vector<std::string> dummyVariablesList;
  std::vector<EventVarTransformLib> eventVarTransformList;

  struct FromHistContent{
    bool isEnabled{false};
    std::string rootFilePath{};

    struct SampleHist{
      std::string name{};
      std::string hist{};
      std::vector<std::string> axisList{};
    };
    std::vector<SampleHist> sampleHistList{};

    SampleHist& addSampleHist(const std::string& name_){
      for( auto& sampleHist : sampleHistList ){
        LogThrowIf(sampleHist.name == name_, "Duplicate sample hist with name: " << name_);
      }
      sampleHistList.emplace_back();
      sampleHistList.back().name = name_;
      return sampleHistList.back();
    }
    SampleHist* getSampleHistPtr(const std::string& name_){
      for( auto& sampleHist : sampleHistList ){
        if( sampleHist.name == name_ ){ return &sampleHist; }
      }
      return nullptr;
    }
  };
  FromHistContent fromHistContent;

//  JsonType fromHistContent{};
  JsonType overridePropagatorConfig{};

  [[nodiscard]] std::string getSummary() const;
};

struct DataDispenserCache{
  Propagator* propagatorPtr{nullptr};

  size_t totalNbEvents{0};

  std::vector<Sample*> samplesToFillList{};
  std::vector<size_t> sampleNbOfEvents;
  std::vector<int> entrySampleIndexList{};
  std::vector<size_t> sampleIndexOffsetList;
  std::vector< std::vector<Event>* > sampleEventListPtrToFill;
  std::vector<DialCollection*> dialCollectionsRefList{};

  std::vector<std::string> varsRequestedForIndexing{};
  std::vector<std::string> varsRequestedForStorage{};
  std::map<std::string, std::pair<std::string, bool>> varToLeafDict; // varToLeafDict[EVENT_VAR_NAME] = {LEAF_NAME, IS_DUMMY}

  std::vector<std::string> varsToOverrideList; // stores the leaves names to override in the right order

  struct ThreadSelectionResult{
    std::vector<size_t> sampleNbOfEvents;
    std::vector<int> entrySampleIndexList;
  };
  std::vector<ThreadSelectionResult> threadSelectionResults;

  void clear();
  void addVarRequestedForIndexing(const std::string& varName_);
  void addVarRequestedForStorage(const std::string& varName_);

};

struct ThreadSharedData{
  Long64_t nbEntries{0};

  const GenericToolbox::LeafForm* eventVariableAsWeightLeafPtr{nullptr};

  // has to be hooked to the TChain
  TTreeFormula* dialIndexTreeFormula{nullptr};
  TTreeFormula* nominalWeightTreeFormula{nullptr};

  std::shared_ptr<TChain> treeChain{nullptr};

  std::vector<const GenericToolbox::LeafForm*> leafFormIndexingList{};
  std::vector<const GenericToolbox::LeafForm*> leafFormStorageList{};

  // thread communication
  GenericToolbox::Atomic<bool> requestReadNextEntry{false};
  GenericToolbox::Atomic<bool> isEntryBufferReady{false};
  GenericToolbox::Atomic<bool> isDoneReading{false};
  GenericToolbox::Atomic<bool> isEventFillerReady{false};
};



#endif //GUNDAM_DATA_DISPENSER_UTILS_H
