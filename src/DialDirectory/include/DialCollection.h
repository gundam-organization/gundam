//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_DIALCOLLECTION_H
#define GUNDAM_DIALCOLLECTION_H

#include "DialInterface.h"
#include "DialBase.h"

#include "GenericToolbox.Wrappers.h"

#include "nlohmann/json.hpp"

#include "vector"
#include "string"


class DialCollection : public JsonBaseClass {

public:
  DialCollection() = default;

  void propagate(int iThread_);


protected:
  void readConfigImpl() override;

  bool initializeNormDialsWithParBinning();
  bool initializeDialsWithDefinition();
  void readGlobals(const nlohmann::json &config_);
  nlohmann::json fetchDialsDefinition(const nlohmann::json &definitionsList_);


private:
  // parameter
  bool _isEnabled_{true};
  bool _enableDialsSummary_{false};
  std::string _applyConditionStr_{};
  std::vector<std::string> _dataSetNameList_{};
  std::shared_ptr<TFormula> _applyConditionFormula_{nullptr};

  DialType::DialType _globalDialType_{DialType::DialType_OVERFLOW};
  std::string _globalDialSubType_{};
  std::string _globalDialLeafName_{};
  double _minDialResponse_{std::nan("unset")};
  double _maxDialResponse_{std::nan("unset")};
  bool _useMirrorDial_{false};
  double _mirrorLowEdge_{std::nan("unset")};
  double _mirrorHighEdge_{std::nan("unset")};
  double _mirrorRange_{std::nan("unset")};

  bool _allowDialExtrapolation_{false};

  std::vector<DataBinSet> _binningCacheList_;

  // internal
  bool _parallelizeEventList_{false};
  std::string _title_{};

  std::vector<DialInterface> _dialsList_{}; // what is going to be stored in each event?
  std::vector<GenericToolbox::PolymorphicObjectWrapper<DialBase>> _dialBaseList_;

};


#endif //GUNDAM_DIALCOLLECTION_H
