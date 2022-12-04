//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_DIALCOLLECTION_H
#define GUNDAM_DIALCOLLECTION_H

#include "DialBase.h"
#include "DialInterface.h"
#include "DialInputBuffer.h"
#include "DialResponseSupervisor.h"
#include "FitSampleSet.h"

#include "GenericToolbox.Wrappers.h"

#include "nlohmann/json.hpp"

#include "vector"
#include "string"
#include "memory"


class DialCollection : public JsonBaseClass {

public:
  DialCollection(FitSampleSet *targetSampleSetPtr, std::vector<FitParameterSet> *targetParameterSetListPtr);

  void setSupervisedParameterIndex(int supervisedParameterIndex);
  void setSupervisedParameterSetIndex(int supervisedParameterSetIndex);

  [[nodiscard]] bool isBinned() const;
  [[nodiscard]] const std::string &getGlobalDialLeafName() const;
  [[nodiscard]] const std::string &getGlobalDialType() const;
  [[nodiscard]] const std::shared_ptr<TFormula> &getApplyConditionFormula() const;
  std::vector<GenericToolbox::PolymorphicObjectWrapper<DialBase>> &getDialBaseList();
  std::vector<DialInterface> &getDialInterfaceList();

  std::string getTitle();
  std::string getSummary(bool shallow_ = true);
  [[nodiscard]] bool isDatasetValid(const std::string& datasetName_) const;
  size_t getNextDialFreeSlot();
  void shrinkContainers();
  void setupDialInterfaceReferences();
  void updateInputBuffers();

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

  bool initializeNormDialsWithParBinning();
  bool initializeDialsWithDefinition();
  void readGlobals(const nlohmann::json &config_);
  nlohmann::json fetchDialsDefinition(const nlohmann::json &definitionsList_);

private:
  // parameters
  bool _isEnabled_{true};
  bool _useMirrorDial_{false};
  bool _enableDialsSummary_{false};
  bool _allowDialExtrapolation_{false};
  bool _isBinned_{true};
  double _minDialResponse_{std::nan("unset")};
  double _maxDialResponse_{std::nan("unset")};
  double _mirrorLowEdge_{std::nan("unset")};
  double _mirrorHighEdge_{std::nan("unset")};
  double _mirrorRange_{std::nan("unset")};
  std::string _applyConditionStr_{};
  std::string _globalDialSubType_{};
  std::string _globalDialLeafName_{};
  std::string _globalDialType_{};
  std::vector<std::string> _dataSetNameList_{};
  std::shared_ptr<TFormula> _applyConditionFormula_{nullptr};
//  DialType::DialType _globalDialType_{ DialType::DialType_OVERFLOW };

  int _supervisedParameterIndex_{-1};
  int _supervisedParameterSetIndex_{-1};

  // internal
//  bool _parallelizeDials_{false}; // instead of event parallelization
  size_t _dialFreeSlot_{0};
  GenericToolbox::NoCopyWrapper<std::mutex> _mutex_{};

  std::vector<DialInterface> _dialInterfaceList_{};
  std::vector<DialInputBuffer> _dialInputBufferList_{};
  std::vector<DialResponseSupervisor> _dialResponseSupervisorList_{};
  std::vector<GenericToolbox::PolymorphicObjectWrapper<DialBase>> _dialBaseList_{};

  // external refs
  FitSampleSet* _sampleSetPtr_{nullptr};
  std::vector<FitParameterSet>* _parameterSetListPtr_{nullptr};

};


#endif //GUNDAM_DIALCOLLECTION_H
