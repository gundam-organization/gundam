//
// Created by Nadrino on 21/05/2021.
//

#ifndef GUNDAM_DIALSET_H
#define GUNDAM_DIALSET_H

#include "Dial.h"

#include "GenericToolbox.h"

#include "TFormula.h"
#include "json.hpp"

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

  void setParameterIndex(int parameterIndex);
  void setParameterName(const std::string &parameterName);
  void setConfig(const nlohmann::json &config_);
  void setWorkingDirectory(const std::string &workingDirectory);
  void setAssociatedParameterReference(FitParameter* associatedParameterReference);
  void setCurrentDialOffset(size_t currentDialOffset);

  void initialize();

  // Getters
  bool isEnabled() const;
  std::vector<std::shared_ptr<Dial>> &getDialList();
  const std::vector<std::string> &getDataSetNameList() const;
  TFormula *getApplyConditionFormula() const;
  const std::string &getDialLeafName() const;
  const std::string &getDialSubType() const;
  const std::string &getParameterName() const;
  size_t getCurrentDialOffset() const;
  DialType::DialType getGlobalDialType() const;
  const Dial &getTemplateDial() const;
  const FitParameter* getOwnerFitParameter() const { return _associatedParameterReference_; }

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
  // Parameters
  nlohmann::json _config_;
  int _parameterIndex_{-1};
  std::string _parameterName_;
  std::string _workingDirectory_{"."};
  std::string _applyConditionStr_;
  std::shared_ptr<TFormula> _applyConditionFormula_{nullptr};
  const FitParameter* _associatedParameterReference_{nullptr};

  // Internals
  bool _enableDialsSummary_{false};
  bool _isEnabled_{true};
  std::vector<std::string> _dataSetNameList_;
//  double _parameterNominalValue_{}; // parameter with which the MC has produced the data set

  // shared pointers are needed since we want to make vectors of DialSets.
  // .emplace_back() method is calling delete which is calling reset(), and this one has to delete the content of
  // every pointers. It means the new copied DialSet will handle Dial ptr which have already been deleted.
  std::vector<std::shared_ptr<Dial>> _dialList_{};
  size_t _currentDialOffset_{0};

  // globals
  DialType::DialType _globalDialType_{DialType::DialType_OVERFLOW};
  std::string _globalDialSubType_{};
  std::string _globalDialLeafName_{};
  double _minDialResponse_{std::nan("unset")};
  double _maxDialResponse_{std::nan("unset")};
  bool _globalUseMirrorDial_{false};
  double _mirrorLowEdge_{std::nan("unset")};
  double _mirrorHighEdge_{std::nan("unset")};
  double _mirrorRange_{std::nan("unset")};

};


#endif //GUNDAM_DIALSET_H
