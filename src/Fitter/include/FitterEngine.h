//
// Created by Nadrino on 11/06/2021.
//

#ifndef GUNDAM_FITTERENGINE_H
#define GUNDAM_FITTERENGINE_H


#include "Propagator.h"
#include "LikelihoodInterface.h"
#include "MinimizerBase.h"
#include "JsonBaseClass.h"

#include "GenericToolbox.VariablesMonitor.h"
#include "GenericToolbox.CycleTimer.h"

#include "TDirectory.h"
#include "Math/Functor.h"
#include "Math/Minimizer.h"
#include "nlohmann/json.hpp"

#include <string>
#include <vector>
#include <memory>


class FitterEngine : public JsonBaseClass {

public:
  explicit FitterEngine(TDirectory *saveDir_);

  // Setters
  void setSaveDir(TDirectory *saveDir);
  void setIsDryRun(bool isDryRun_);
  void setEnablePca(bool enablePca_);
  void setEnablePreFitScan(bool enablePreFitScan);
  void setEnablePostFitScan(bool enablePostFitScan);
  void setEnablePreFitToPostFitLineScan(bool enablePreFitToPostFitScan);
  void setGenerateSamplePlots(bool generateSamplePlots);
  void setGenerateOneSigmaPlots(bool generateOneSigmaPlots);
  void setDoAllParamVariations(bool doAllParamVariations_);
  void setAllParamVariationsSigmas(const std::vector<double> &allParamVariationsSigmas);
  void setThrowMcBeforeFit(bool throwMcBeforeFit_){ _throwMcBeforeFit_ = throwMcBeforeFit_; }
  void setThrowGain(double throwGain_){ _throwGain_ = throwGain_; }

  // Getters (const)
  const nlohmann::json &getPreFitParState() const;
  const nlohmann::json &getPostFitParState() const;
  [[nodiscard]] const Propagator& getPropagator() const;
  [[nodiscard]] const MinimizerBase& getMinimizer() const;
  [[nodiscard]] const LikelihoodInterface& getLikelihood() const;

  // Getters (non-const)
  Propagator& getPropagator();
  MinimizerBase& getMinimizer();
  LikelihoodInterface& getLikelihood();
  TDirectory* getSaveDir();

  // Core
  void fit();

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

  void fixGhostFitParameters();
  void rescaleParametersStepSize();

  // Scan the parameters as used by the minimizer (e.g. MINUIT).  This has
  // been replaced by MinimizerBase::scanParameters() which can be accessed
  // through the getMinimizer() method.  For example:
  // "fitter.scanMinimizerParameters(dir)" should be replaced by "fitter.getMinimizer().scanParameters(dir)"
  [[deprecated("Use getMinimizer().scanParameters(dir) instead")]]
       void scanMinimizerParameters(TDirectory* saveDir_);
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
  std::vector<double> _allParamVariationsSigmas_{};

  // Internals
  TDirectory* _saveDir_{nullptr};
  Propagator _propagator_{};
  std::unique_ptr<MinimizerBase> _minimizer_{nullptr};
  LikelihoodInterface _likelihood_{this};
  nlohmann::json _preFitParState_{};
  nlohmann::json _postFitParState_{};

};
#endif //GUNDAM_FITTERENGINE_H
