//
// Created by Nadrino on 11/06/2021.
//

#ifndef GUNDAM_FITTERENGINE_H
#define GUNDAM_FITTERENGINE_H


#include "LikelihoodInterface.h"
#include "ParameterScanner.h"
#include "MinimizerBase.h"
#include "JsonBaseClass.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Time.h"

#include "TDirectory.h"
#include "nlohmann/json.hpp"

#include <string>
#include <vector>
#include <memory>


class FitterEngine : public JsonBaseClass {

#define ENUM_NAME MinimizerType
#define ENUM_FIELDS \
  ENUM_FIELD( RootFactory, 0 ) \
  ENUM_FIELD( SimpleMCMC )
#include "GenericToolbox.MakeEnum.h"

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

public:
  FitterEngine() = default;

  // Setters
  void setSaveDir(TDirectory *saveDir){ _saveDir_ = saveDir; }
  void setIsDryRun(bool isDryRun_){ _isDryRun_ = isDryRun_; }
  void setEnablePca(bool enablePca_){ _enablePca_ = enablePca_; }
  void setEnablePreFitScan(bool enablePreFitScan){ _enablePreFitScan_ = enablePreFitScan; }
  void setEnablePostFitScan(bool enablePostFitScan){ _enablePostFitScan_ = enablePostFitScan; }
  void setEnablePreFitToPostFitLineScan(bool enablePreFitToPostFitScan){ _enablePreFitToPostFitLineScan_ = enablePreFitToPostFitScan; }
  void setGenerateSamplePlots(bool generateSamplePlots){ _generateSamplePlots_ = generateSamplePlots; }
  void setGenerateOneSigmaPlots(bool generateOneSigmaPlots){ _generateOneSigmaPlots_ = generateOneSigmaPlots; }
  void setDoAllParamVariations(bool doAllParamVariations_){ _doAllParamVariations_ = doAllParamVariations_; }
  void setAllParamVariationsSigmas(const std::vector<double> &allParamVariationsSigmas){ _allParamVariationsSigmas_ = allParamVariationsSigmas; }
  void setThrowMcBeforeFit(bool throwMcBeforeFit_){ _throwMcBeforeFit_ = throwMcBeforeFit_; }
  void setThrowGain(double throwGain_){ _throwGain_ = throwGain_; }

  // Getters (const)
  const JsonType &getPreFitParState() const{ return _preFitParState_; }
  const JsonType &getPostFitParState() const{ return _postFitParState_; }
  [[nodiscard]] const Propagator& getPropagator() const{ return _likelihoodInterface_.getPropagator(); }
  [[nodiscard]] const ParameterScanner& getParameterScanner() const{ return _parameterScanner_; }
  [[nodiscard]] const MinimizerBase& getMinimizer() const{ return *_minimizer_; }
  [[nodiscard]] const LikelihoodInterface& getLikelihoodInterface() const{ return _likelihoodInterface_; }

  // Getters (non-const)
  Propagator& getPropagator(){ return _likelihoodInterface_.getPropagator(); }
  ParameterScanner& getParameterScanner(){ return _parameterScanner_; }
  MinimizerBase& getMinimizer(){ return *_minimizer_; }
  LikelihoodInterface& getLikelihoodInterface(){ return _likelihoodInterface_; }
  TDirectory* getSaveDir(){ return _saveDir_; }

  // Core
  void fit();

protected:
  void runPcaCheck();
  void rescaleParametersStepSize();
  void checkNumericalAccuracy();

private:
  // Parameters
  bool _isDryRun_{false};
  bool _enablePca_{false};
  bool _throwMcBeforeFit_{false};
  bool _enablePreFitScan_{false};
  bool _enablePostFitScan_{false};
  bool _enablePreFitToPostFitLineScan_{true};
  bool _generateSamplePlots_{true};
  bool _generateOneSigmaPlots_{false};
  bool _doAllParamVariations_{false};
  bool _scaleParStepWithChi2Response_{false};
  double _throwGain_{1.};
  double _parStepGain_{0.1};
  double _pcaDeltaChi2Threshold_{1E-6};
  bool _savePostfitEventTrees_{false};
  std::vector<double> _allParamVariationsSigmas_{};
  JsonType _preFitParState_{};
  JsonType _postFitParState_{};

  // Internals
  TDirectory* _saveDir_{nullptr};
  LikelihoodInterface _likelihoodInterface_{};
  ParameterScanner _parameterScanner_{};
  MinimizerType _minimizerType_{};
  std::unique_ptr<MinimizerBase> _minimizer_{}; // a virtual class in charge of driving the LikelihoodInterface

};
#endif //GUNDAM_FITTERENGINE_H
