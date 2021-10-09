//
// Created by Nadrino on 21/04/2021.
//

#include "memory"

#include "TFile.h"
#include "json.hpp"

// Globals
class FlatTreeConverter {

public:
  FlatTreeConverter() = default;
  virtual ~FlatTreeConverter() = default;

  void loadConfig(const nlohmann::json &config_);

private:
  nlohmann::json _config_;
  std::shared_ptr<TFile> outputFilePtr;

};
