//
// Created by Adrien BLANCHET on 22/03/2022.
//

#include "NestedDialTest.h"

#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[NestedDial]");
})

NestedDialTest::NestedDialTest() = default;
NestedDialTest::~NestedDialTest() = default;

void NestedDialTest::setConfig(const nlohmann::json &config_) {
  _config_ = config_;
}

void NestedDialTest::setName(const std::string &name_) {
  _name_ = name_;
}
void NestedDialTest::setFormulaStr(const std::string& formulaStr_){
  _evalFormula_ = TFormula(formulaStr_.c_str(), formulaStr_.c_str());
  LogThrowIf(not _evalFormula_.IsValid(), "\"" << formulaStr_ << "\": could not be parsed as formula expression.")
  _dialResponsesCache_.resize(_evalFormula_.GetNdim(), std::nan("unset"));
}
void NestedDialTest::setApplyConditionStr(const std::string& applyConditionStr_){
  _applyConditionFormula_ = TFormula(applyConditionStr_.c_str(), applyConditionStr_.c_str());
  LogThrowIf(not _applyConditionFormula_.IsValid(), "\"" << applyConditionStr_ << "\": could not be parsed as formula expression.")
}

void NestedDialTest::initialize() {
  this->readConfig();
  LogThrowIf(not _evalFormula_.IsValid(), "\"" << _evalFormula_.GetTitle() << "\": could not be parsed as formula expression.")
}


double NestedDialTest::eval(const std::vector<Dial*>& dialRefList_) {
  this->updateDialResponseCache(dialRefList_);
  return _evalFormula_.EvalPar(&_dialResponsesCache_[0]);
}

void NestedDialTest::readConfig(){
  if( _config_.empty() ) return;
  this->setName(JsonUtils::fetchValue(_config_, "name", ""));
  this->setApplyConditionStr(JsonUtils::fetchValue<std::string>(_config_, "applyCondition"));
  this->setFormulaStr(JsonUtils::fetchValue<std::string>(_config_, "evalFormula"));
}
void NestedDialTest::updateDialResponseCache(const std::vector<Dial*>& dialRefList_) {
  auto dialIt = dialRefList_.begin();
  auto valIt = _dialResponsesCache_.begin();
  for( ; dialIt != dialRefList_.end() && valIt != _dialResponsesCache_.end(); ++dialIt, ++valIt){
    (*valIt) = (*dialIt)->evalResponse();
  }
}

