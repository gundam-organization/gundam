//
// Created by Adrien BLANCHET on 14/05/2022.
//

#ifndef GUNDAM_DATA_DISPENSER_H
#define GUNDAM_DATA_DISPENSER_H

#include "EventVarTransformLib.h"
#include "DataDispenserUtils.h"

#include "Propagator.h"
#include "JsonBaseClass.h"

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
  std::unique_ptr<TChain> openChain(bool verbose_ = false);

  // multi-thread
  void eventSelectionFunction(int iThread_);
  void fillFunction(int iThread_);


private:
  // config
  DataDispenserParameters _parameters_;

  // internals
  DatasetLoader* _owner_{nullptr};
  DataDispenserCache _cache_;

};


#endif //GUNDAM_DATA_DISPENSER_H
