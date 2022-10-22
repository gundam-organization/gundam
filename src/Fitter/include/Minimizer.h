//
// Created by Adrien BLANCHET on 16/12/2021.
//

#ifndef GUNDAM_MINIMIZER_H
#define GUNDAM_MINIMIZER_H


#include "FitParameterSet.h"

#include "Math/Minimizer.h"
#include "Math/Functor.h"
#include "nlohmann/json.hpp"

#include "memory"
#include "vector"

class Minimizer {

public:
  Minimizer();
  virtual ~Minimizer();

  void reset();

  void setConfig(const nlohmann::json &config);
  void setFitterEnginePtr(void *fitterEnginePtr);

  void initialize();

  bool isIsInitialized() const;
  bool isMinimizeSucceeded() const;

  void minimize();

protected:
  void fillNbFitParameters();
  void defineFitParameters();

private:
  bool _isInitialized_{false};
  nlohmann::json _config_;
  bool _useNormalizedFitSpace_{true};
  std::string _minimizerType_;
  std::string _minimizerAlgo_;

  // external references
  void* _fitterEnginePtr_{nullptr};
  const std::vector<FitParameterSet>* _parSetListPtr_{nullptr};

  int _nbFitParameters_{};
  std::shared_ptr<ROOT::Math::Minimizer> _minimizer_{nullptr};
  ROOT::Math::Functor _llhEvalFunction_;

  // results
  bool _minimizeSucceeded_{false};
};


#endif //GUNDAM_MINIMIZER_H
