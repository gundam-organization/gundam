//
// Created by Adrien BLANCHET on 14/05/2022.
//

#ifndef GUNDAM_DATADISPENSER_H
#define GUNDAM_DATADISPENSER_H

#include "EventVarTransformLib.h"
#include "FitSampleSet.h"
#ifndef USE_NEW_DIALS
#include "DialSet.h"
#endif
#include "FitParameterSet.h"
#include "PlotGenerator.h"
#include "JsonBaseClass.h"
#include "DialCollection.h"
#include "EventDialCache.h"

#include "TChain.h"

#include "string"
#include "vector"
#include "map"

class DatasetLoader;

struct DataDispenserParameters{
  bool useMcContainer{false}; // define the container to fill -> could get rid of it?
  std::string name{};
  std::string treePath{};
  std::string nominalWeightFormulaStr{};
  std::string selectionCutFormulaStr{};
  std::vector<std::string> activeLeafNameList{};
  std::vector<std::string> filePathList{};
  std::map<std::string, std::string> overrideLeafDict{};
  std::vector<std::string> additionalVarsStorage{};
  std::vector<std::string> dummyVariablesList;
  int iThrow{-1};

  [[nodiscard]] std::string getSummary() const{
    std::stringstream ss;
    ss << GET_VAR_NAME_VALUE(useMcContainer);
    ss << std::endl << GET_VAR_NAME_VALUE(name);
    ss << std::endl << GET_VAR_NAME_VALUE(treePath);
    ss << std::endl << GET_VAR_NAME_VALUE(nominalWeightFormulaStr);
    ss << std::endl << GET_VAR_NAME_VALUE(selectionCutFormulaStr);
    ss << std::endl << "activeLeafNameList = " << GenericToolbox::parseVectorAsString(activeLeafNameList, true);
    ss << std::endl << "filePathList = " << GenericToolbox::parseVectorAsString(filePathList, true);
    ss << std::endl << "overrideLeafDict = " << GenericToolbox::parseMapAsString(overrideLeafDict, true);
    ss << std::endl << "additionalVarsStorage = " << GenericToolbox::parseVectorAsString(additionalVarsStorage, true);
    ss << std::endl << GET_VAR_NAME_VALUE(iThrow);
    return ss.str();
  }
};
struct DataDispenserCache{
  std::vector<FitSample*> samplesToFillList{};
  std::vector<size_t> sampleNbOfEvents;
  std::vector<std::vector<bool>> eventIsInSamplesList{};
  std::vector<GenericToolbox::CopiableAtomic<size_t>> sampleIndexOffsetList;
  std::vector< std::vector<PhysicsEvent>* > sampleEventListPtrToFill;
#ifndef USE_NEW_DIALS
  std::map<FitParameterSet*, std::vector<DialSet*>> dialSetPtrMap;
#endif
  std::vector<DialCollection*> dialCollectionsRefList{};

  std::vector<std::string> varsRequestedForIndexing{};
  std::vector<std::string> varsRequestedForStorage{};
  std::map<std::string, std::pair<std::string, bool>> varToLeafDict; // varToLeafDict[EVENT_VAR_NAME] = {LEAF_NAME, IS_DUMMY}

  std::vector<std::string> varsToOverrideList; // stores the leaves names to override in the right order

  // Variable transformations
  std::vector<EventVarTransformLib> eventVarTransformList;

  void clear(){
    samplesToFillList.clear();
    sampleNbOfEvents.clear();
    eventIsInSamplesList.clear();

    sampleIndexOffsetList.clear();
    sampleEventListPtrToFill.clear();
#ifndef USE_NEW_DIALS
    dialSetPtrMap.clear();
#endif

    varsRequestedForIndexing.clear();
    varsRequestedForStorage.clear();
    varToLeafDict.clear();

    varsToOverrideList.clear();

    eventVarTransformList.clear();
  }
  void addVarRequestedForIndexing(const std::string& varName_);
  void addVarRequestedForStorage(const std::string& varName_);

};

class DataDispenser : public JsonBaseClass {

public:
  explicit DataDispenser(DatasetLoader* owner_);
  void setOwner(DatasetLoader* owner_){ _owner_ = owner_; }

  [[nodiscard]] const DataDispenserParameters &getParameters() const;
  DataDispenserParameters &getParameters();

  void setSampleSetPtrToLoad(FitSampleSet *sampleSetPtrToLoad);
  void setParSetPtrToLoad(std::vector<FitParameterSet> *parSetListPtrToLoad_);
  void setDialCollectionListPtr(std::vector<DialCollection> *dialCollectionListPtr);
  void setPlotGenPtr(PlotGenerator *plotGenPtr);
  void setEventDialCache(EventDialCache* eventDialCache_);

  std::string getTitle();

  void load();
  GenericToolbox::TreeEntryBuffer generateTreeEventBuffer(TChain* treeChain_, const std::vector<std::string>& varsList_);

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

  void buildSampleToFillList();
  void doEventSelection();
  void fetchRequestedLeaves();
  void preAllocateMemory();
  void readAndFill();

  void fillFunction(int iThread_);

private:
  // Parameters
  DataDispenserParameters _parameters_;

  // Internals
  DatasetLoader* _owner_{nullptr};
  FitSampleSet* _sampleSetPtrToLoad_{nullptr};
  std::vector<FitParameterSet>* _parSetListPtrToLoad_{nullptr};
  std::vector<DialCollection>* _dialCollectionListPtr_{nullptr};
  PlotGenerator* _plotGenPtr_{nullptr}; // used to know which vars have to be kept in memory
  EventDialCache* _eventDialCacheRef_{nullptr};

  DataDispenserCache _cache_;

};


#endif //GUNDAM_DATADISPENSER_H
