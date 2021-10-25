//
// Created by Nadrino on 21/05/2021.
//

#ifndef XSLLHFITTER_DIALSET_H
#define XSLLHFITTER_DIALSET_H

#include "string"
#include "vector"
#include "json.hpp"
#include "memory"

#include "TFormula.h"

#include "GenericToolbox.h"

#include "Dial.h"


class DialSet {

public:
  DialSet();
  virtual ~DialSet();

  void reset();

  void setParameterIndex(int parameterIndex);
  void setParameterName(const std::string &parameterName);
  void setDialSetConfig(const nlohmann::json &dialSetConfig);
  void setWorkingDirectory(const std::string &workingDirectory);

  void setAssociatedParameterReference(void *associatedParameterReference);

  void initialize();

  // Getters
  bool isEnabled() const;
  std::vector<std::shared_ptr<Dial>> &getDialList();
  const std::vector<std::string> &getDataSetNameList() const;
  TFormula *getApplyConditionFormula() const;
  void *getAssociatedParameterReference();
  const std::string &getDialLeafName() const;


  // Core
  std::string getSummary() const;

protected:
  bool initializeNormDialsWithBinning();
  bool initializeDialsWithDefinition();
  nlohmann::json fetchDialsDefinition(const nlohmann::json &definitionsList_);

private:
  // Parameters
  nlohmann::json _dialSetConfig_;
  int _parameterIndex_{-1};
  std::string _parameterName_;
  std::string _workingDirectory_{"."};
  std::string _applyConditionStr_;
  TFormula* _applyConditionFormula_{nullptr};
  void* _associatedParameterReference_{nullptr};
  std::string _dialLeafName_{};

  // Internals
  bool _enableDialsSummary_{false};
  bool _isEnabled_{true};
  double _parameterNominalValue_{}; // parameter with which the MC has produced the data set
  std::vector<std::string> _dataSetNameList_;
  // shared pointers are needed since we want to make vectors of DialSets.
  // .emplace_back() method is calling delete which is calling reset(), and this one has to delete the content of
  // every pointers. It means the new copied DialSet will handle Dial ptr which have already been deleted.
  std::vector<std::shared_ptr<Dial>> _dialList_;
  DialType::DialType _globalDialType_;

public:
  static bool _verboseMode_;

};


#endif //XSLLHFITTER_DIALSET_H
