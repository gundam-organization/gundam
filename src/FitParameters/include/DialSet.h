//
// Created by Nadrino on 21/05/2021.
//

#ifndef GUNDAM_DIALSET_H
#define GUNDAM_DIALSET_H

#include "DialWrapper.h"
#include "DataBinSet.h"
#include "GlobalVariables.h"

#include "GenericToolbox.h"

#include "TFormula.h"
#include "nlohmann/json.hpp"

#include "string"
#include "vector"
#include "memory"


class FitParameter;

class DialSet {

public:
  static bool _verboseMode_;

public:
  DialSet();
  virtual ~DialSet();

  void reset();

  void setOwner(const FitParameter* owner_);
  void setConfig(const nlohmann::json &config_);

  void initialize();

  // Getters
  bool isEnabled() const;
  std::vector<DialWrapper> &getDialList();
  const std::vector<std::string> &getDataSetNameList() const;
  TFormula *getApplyConditionFormula() const;
  const std::string &getDialLeafName() const;
  DialType::DialType getGlobalDialType() const;
  const FitParameter* getOwner() const { return _owner_; }

  double getMinDialResponse() const;
  double getMaxDialResponse() const;
  bool useMirrorDial() const;
  double getMirrorLowEdge() const;
  double getMirrorHighEdge() const;
  double getMirrorRange() const;

  // Core
  std::string getSummary() const;
  void applyGlobalParameters(Dial* dial_) const;
  void applyGlobalParameters(Dial& dial_) const;

protected:
  void readGlobals(const nlohmann::json &config_);
  bool initializeNormDialsWithParBinning();
  bool initializeDialsWithDefinition();
  nlohmann::json fetchDialsDefinition(const nlohmann::json &definitionsList_);

private:
  // owner
  const FitParameter* _owner_{nullptr};

  // Parameters
  nlohmann::json _config_;
  bool _isEnabled_{true};
  std::string _applyConditionStr_;
  std::shared_ptr<TFormula> _applyConditionFormula_{nullptr};

  // Internals
  bool _enableDialsSummary_{false};
  std::vector<std::string> _dataSetNameList_;

//  std::vector<DialWrapper<Dial>> _dialList_{};
  std::vector<DialWrapper> _dialList_{};

  // globals
  DialType::DialType _globalDialType_{DialType::DialType_OVERFLOW};
  std::string _globalDialLeafName_{};
  double _minDialResponse_{std::nan("unset")};
  double _maxDialResponse_{std::nan("unset")};
  bool _useMirrorDial_{false};
  double _mirrorLowEdge_{std::nan("unset")};
  double _mirrorHighEdge_{std::nan("unset")};
  double _mirrorRange_{std::nan("unset")};

  std::vector<DataBinSet> _binningCacheList_;

};


#endif //GUNDAM_DIALSET_H
