//
// Created by Adrien BLANCHET on 14/05/2022.
//

#ifndef GUNDAM_DATA_DISPENSER_H
#define GUNDAM_DATA_DISPENSER_H

#include "EventVarTransformLib.h"
#include "DataDispenserUtils.h"

#include "PlotGenerator.h"
#include "Propagator.h"

#include "GenericToolbox.Thread.h"

#include "TChain.h"

#include <map>
#include <string>
#include <vector>


class DatasetDefinition; // owner


class DataDispenser : public JsonBaseClass {

protected:
  void configureImpl() override;
  void initializeImpl() override;

public:
  DataDispenser() = delete; // owner should be set
  explicit DataDispenser( DatasetDefinition* owner_): _owner_(owner_) {}

  // setters
  void setOwner( DatasetDefinition* owner_){ _owner_ = owner_; }
  void setPlotGeneratorPtr( const PlotGenerator* plotGeneratorPtr_ ){ _plotGeneratorPtr_ = plotGeneratorPtr_; }

  // const getters
  [[nodiscard]] const DatasetDefinition* getOwner() const{ return _owner_; }
  [[nodiscard]] const DataDispenserParameters &getParameters() const{ return _parameters_; }

  // non-const getters
  DataDispenserParameters &getParameters(){ return _parameters_; }

  // misc
  std::string getTitle();

  // core
  void load(Propagator& propagator_);

protected:
  void buildSampleToFillList();
  void parseStringParameters();
  void doEventSelection();
  void fetchRequestedLeaves();
  void preAllocateMemory();
  void readAndFill();
  void loadFromHistContent();

  // utils
  std::shared_ptr<TChain> openChain(bool verbose_ = false);

  // multi-thread
  void eventSelectionFunction(int iThread_);

  void runEventFillThreads(int iThread_);
  void loadEvent(int iThread_);

  struct ThreadSharedData{
    Long64_t nbEntries{0};

    std::shared_ptr<TChain> treeChain{nullptr};

    std::vector<const GenericToolbox::LeafForm*> leafFormIndexingList{};
    std::vector<const GenericToolbox::LeafForm*> leafFormStorageList{};

    // has to be hooked to the TChain
    TTreeFormula* dialIndexTreeFormula{nullptr};
    TTreeFormula* nominalWeightTreeFormula{nullptr};

    // thread communication
    GenericToolbox::Atomic<bool> requestReadNextEntry{false};
    GenericToolbox::Atomic<bool> isEntryBufferReady{false};
    GenericToolbox::Atomic<bool> isDoneReading{false};
    GenericToolbox::Atomic<bool> isEventFillerReady{false};
  };
  std::vector<ThreadSharedData> threadSharedDataList{};


private:
  // config
  DataDispenserParameters _parameters_;

  // internals
  DatasetDefinition* _owner_{nullptr};
  DataDispenserCache _cache_;

  // needed to check which variables need to be loaded
  const PlotGenerator* _plotGeneratorPtr_{nullptr};

  // multi-threading
  GenericToolbox::ParallelWorker _threadPool_{};
  GenericToolbox::ParallelWorker _threadPoolEventLoad_{};
  GenericToolbox::NoCopyWrapper<std::mutex> _mutex_{};

};


#endif //GUNDAM_DATA_DISPENSER_H
