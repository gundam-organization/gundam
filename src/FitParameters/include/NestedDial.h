//
// Created by Adrien BLANCHET on 22/03/2022.
//

#ifndef GUNDAM_NESTEDDIAL_H
#define GUNDAM_NESTEDDIAL_H

#include "Dial.h"

#include "JsonUtils.h"

#include "json.hpp"
#include "TFormula.h"

#include "vector"
#include "string"

class NestedDial {

public:
  NestedDial();
  virtual ~NestedDial();

  void setConfig(const nlohmann::json& config_);

  void setName(const std::string &name_);
  void setFormulaStr(const std::string& formulaStr_);
  void setApplyConditionStr(const std::string& applyConditionStr_);

  void initialize();

  double eval(const std::vector<Dial*>& dialRefList_);

protected:
  void readConfig();
  void updateDialResponseCache(const std::vector<Dial*>& dialRefList_);

private:
  nlohmann::json _config_{};
  std::string _name_{};

  std::vector<Dial*> _dialRefList_{};
  TFormula _evalFormula_{};
  TFormula _applyConditionFormula_{};

  std::vector<double> _dialResponsesCache_{};

};


#endif //GUNDAM_NESTEDDIAL_H
