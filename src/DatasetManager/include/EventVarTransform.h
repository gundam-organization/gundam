//
// Created by Adrien BLANCHET on 13/10/2022.
//

#ifndef GUNDAM_EVENTVARTRANSFORM_H
#define GUNDAM_EVENTVARTRANSFORM_H

#include "PhysicsEvent.h"
#include "JsonBaseClass.h"

#include "TFormula.h"

#include <vector>
#include <string>
#include <memory>


class EventVarTransform : public JsonBaseClass {

public:
  EventVarTransform() = default;
  explicit EventVarTransform(const JsonType& config_);

  void setIsEnabled(bool isEnabled_){ _isEnabled_=isEnabled_; }
  void setIndex(int index_){ _index_ = index_; }
  void setUseCache(bool useCache_){ _useCache_ = useCache_; }

  bool isEnabled(){ return _isEnabled_; }
  bool useCache() const { return _useCache_; }
  int getIndex() const { return _index_; }
  const std::string &getName() const { return _name_; }
  const std::string &getOutputVariableName() const { return _outputVariableName_; }
  const std::vector<std::string>& fetchRequestedVars() const;

  double eval(const PhysicsEvent& event_);
  void storeCachedOutput(PhysicsEvent& event_) const;
  void storeOutput(double output_, PhysicsEvent& storeEvent_) const;
  void evalAndStore(PhysicsEvent& event_);
  void evalAndStore(const PhysicsEvent& evalEvent_, PhysicsEvent& storeEvent_);

protected:
  void initializeImpl() override;
  void readConfigImpl() override;

  double evalTransformation(const PhysicsEvent& event_) const;
  virtual double evalTransformation(const PhysicsEvent& event_, std::vector<double>& inputBuffer_) const;

  // config
  bool _isEnabled_{true};
  std::string _name_;
  std::string _messageOnError_;
  std::string _outputVariableName_;
  std::vector<std::string> _inputFormulaStrList_;

  // Parameters
  int _index_{-1};

  // Internals
  bool _useCache_{true};
  std::vector<TFormula> _inputFormulaList_;

  // CACHES / not parallelisable
  double _outputCache_{};
  std::vector<double> _inputBuffer_;
  mutable std::vector<std::string> _requestedLeavesForEvalCache_{};

};


#endif //GUNDAM_EVENTVARTRANSFORM_H
