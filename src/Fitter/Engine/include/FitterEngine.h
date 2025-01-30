//
// Created by Nadrino on 11/06/2021.
//

#ifndef GUNDAM_FITTER_ENGINE_H
#define GUNDAM_FITTER_ENGINE_H


#include "Propagator.h"
#include "LikelihoodInterface.h"
#include "MinimizerBase.h"


#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Time.h"

#include "TDirectory.h"
#include "Math/Functor.h"
#include "Math/Minimizer.h"

#include <string>
#include <vector>
#include <memory>


class FitterEngine : public JsonBaseClass {

public:
#define ENUM_NAME MinimizerType
#define ENUM_FIELDS \
  ENUM_FIELD( RootMinimizer, 0 ) \
  ENUM_FIELD( AdaptiveMCMC )
#include "GenericToolbox.MakeEnum.h"

#define ENUM_NAME PcaMethod
#define ENUM_FIELDS \
  ENUM_FIELD( DeltaChi2Threshold, 0 ) \
  ENUM_FIELD( ReducedDeltaChi2Threshold ) \
  ENUM_FIELD( SqrtReducedDeltaChi2Threshold )
#include "GenericToolbox.MakeEnum.h"

protected:
  void configureImpl() override;
  void initializeImpl() override;

public:
  explicit FitterEngine(TDirectory *saveDir_) : _saveDir_(saveDir_) {};

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
  void setPcaThreshold(double pcaThreshold_){ _pcaThreshold_ = pcaThreshold_; }
  void setPcaMethod(PcaMethod pcaMethod_){ _pcaMethod_ = pcaMethod_; }

  // const-getters
  [[nodiscard]] const auto& getPreFitParState() const{ return _preFitParState_; }
  [[nodiscard]] const auto& getPostFitParState() const{ return _postFitParState_; }
  [[nodiscard]] MinimizerType getMinimizerType() const{ return _minimizerType_; }
  [[nodiscard]] const MinimizerBase& getMinimizer() const{ return *_minimizer_; }
  [[nodiscard]] const LikelihoodInterface& getLikelihoodInterface() const{ return _likelihoodInterface_; }
  [[nodiscard]] const ParameterScanner& getParameterScanner() const{ return _parameterScanner_; }

  // mutable-getters
  MinimizerBase& getMinimizer(){ return *_minimizer_; }
  LikelihoodInterface& getLikelihoodInterface(){ return _likelihoodInterface_; }
  ParameterScanner& getParameterScanner(){ return _parameterScanner_; }
  TDirectory* getSaveDir(){ return _saveDir_; }

  // Core
  void fit();
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
  bool _savePostfitEventTrees_{false};
  std::vector<double> _allParamVariationsSigmas_{};
  GenericToolbox::Json::JsonType _preFitParState_{};
  GenericToolbox::Json::JsonType _postFitParState_{};

  // dev
  double _pcaThreshold_{0};
  PcaMethod _pcaMethod_{PcaMethod::DeltaChi2Threshold};

  // Internals
  TDirectory* _saveDir_{nullptr};
  LikelihoodInterface _likelihoodInterface_{};
  ParameterScanner _parameterScanner_{};
  MinimizerType _minimizerType_{};
  std::unique_ptr<MinimizerBase> _minimizer_{}; // a virtual class in charge of driving the LikelihoodInterface

};
#endif //GUNDAM_FITTER_ENGINE_H
