//
// Created by Adrien BLANCHET on 14/05/2022.
//

#ifndef GUNDAM_DATADISPENSER_H
#define GUNDAM_DATADISPENSER_H

#include "SampleSet.h"
#include "PlotGenerator.h"
#include "JsonBaseClass.h"
#include "DialCollection.h"
#include "EventDialCache.h"
#include "ParameterSet.h"
#include "EventVarTransformLib.h"
#include "DataDispenserUtils.h"

#include "TChain.h"
#include "nlohmann/json.hpp"

#include <map>
#include <string>
#include <vector>


class DatasetLoader; // owner


class DataDispenser : public JsonBaseClass {

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

public:
  DataDispenser() = delete; // owner should be set
  explicit DataDispenser(DatasetLoader* owner_): _owner_(owner_) {}

  // setters
  void setOwner(DatasetLoader* owner_){ _owner_ = owner_; }

  // const getters
  [[nodiscard]] const DataDispenserParameters &getParameters() const{ return _parameters_; }

  // non-const getters
  DataDispenserParameters &getParameters(){ return _parameters_; }

  void setSampleSetPtrToLoad(SampleSet *sampleSetPtrToLoad);
  void setParSetPtrToLoad(std::vector<ParameterSet> *parSetListPtrToLoad_);
  void setDialCollectionListPtr(std::vector<DialCollection> *dialCollectionListPtr);
  void setPlotGenPtr(PlotGenerator *plotGenPtr);
  void setEventDialCache(EventDialCache* eventDialCache_);

  std::string getTitle();

  void load();

protected:
  void buildSampleToFillList();
  void parseStringParameters();
  void doEventSelection();
  void fetchRequestedLeaves();
  void preAllocateMemory();
  void readAndFill();
  void loadFromHistContent();

  // utils
  std::unique_ptr<TChain> openChain(bool verbose_ = false);

  // multi-thread
  void eventSelectionFunction(int iThread_);
  void fillFunction(int iThread_);


private:
  // config
  DataDispenserParameters _parameters_;

  // internals
  DatasetLoader* _owner_{nullptr};
  SampleSet* _sampleSetPtrToLoad_{nullptr};
  PlotGenerator* _plotGenPtr_{nullptr}; // used to know which vars have to be kept in memory
  EventDialCache* _eventDialCacheRef_{nullptr};
  DataDispenserCache _cache_;
  std::vector<ParameterSet>* _parSetListPtrToLoad_{nullptr};
  std::vector<DialCollection>* _dialCollectionListPtr_{nullptr};

};


#endif //GUNDAM_DATADISPENSER_H
