//
// Created by Adrien BLANCHET on 13/10/2022.
//

#ifndef GUNDAM_EVENTVARTRANSFORM_H
#define GUNDAM_EVENTVARTRANSFORM_H

#include "PhysicsEvent.h"
#include "JsonUtils.h"

#include "TFormula.h"

#include "vector"
#include "string"
#include "memory"


class EventVarTransform {

public:
  EventVarTransform() = default;
  explicit EventVarTransform(const nlohmann::json& config_);

  void initialize(const nlohmann::json& config_);

  void setIndex(int index_);
  void setUseCache(bool useCache_);

  int getIndex() const;
  bool useCache() const;
  const std::string &getOutputVariableName() const;

  const std::vector<std::string>& fetchRequestedVars() const;

  double eval(const PhysicsEvent& event_);
  void storeCachedOutput(PhysicsEvent& event_);
  void storeOutput(double output_, PhysicsEvent& storeEvent_) const;
  void evalAndStore(PhysicsEvent& event_);
  void evalAndStore(const PhysicsEvent& evalEvent_, PhysicsEvent& storeEvent_);

protected:
  void readConfig(const nlohmann::json& config_);
  void loadLibrary();
  void initInputFormulas();

  double evalTransformation(const PhysicsEvent& event_) const;
  double evalTransformation(const PhysicsEvent& event_, std::vector<double>& inputBuffer_) const;

private:
  // Config
  std::string _title_;
  std::string _libraryFile_;
  std::string _messageOnError_;
  std::string _outputVariableName_;
  std::vector<std::string> _inputFormulaStrList_;

  // Parameters
  int _index_{-1};

  // Internals
  bool _useCache_{true};
  std::vector<TFormula> _inputFormulaList_;

  void* _loadedLibrary_{nullptr};
  void* _evalVariable_{nullptr};

  // CACHES / not parallelisable
  double _outputCache_{};
  std::vector<double> _inputBuffer_;
  mutable std::vector<std::string> _requestedLeavesForEvalCache_{};

};


#endif //GUNDAM_EVENTVARTRANSFORM_H
