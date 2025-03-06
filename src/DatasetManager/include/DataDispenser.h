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
  [[nodiscard]] auto* getOwner() const{ return _owner_; }
  [[nodiscard]] auto& getParameters() const{ return _parameters_; }

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
  [[nodiscard]] int getNbParallelCpu() const;
  [[nodiscard]] const std::string& getVariableExpression(const std::string& variable_) const;
  [[nodiscard]] std::shared_ptr<TChain> openChain(bool verbose_ = false) const;

  // multi-thread
  void eventSelectionFunction(int iThread_);

  void runEventFillThreads(int iThread_);
  void loadEvent(int iThread_);

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
  GenericToolbox::NoCopyWrapper<std::mutex> _mutex_{};

};


#endif //GUNDAM_DATA_DISPENSER_H
