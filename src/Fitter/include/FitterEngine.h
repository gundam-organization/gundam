//
// Created by Nadrino on 11/06/2021.
//

#ifndef GUNDAM_FITTERENGINE_H
#define GUNDAM_FITTERENGINE_H


#include "Propagator.h"
#include "MinimizerInterface.h"
#include "JsonBaseClass.h"
#include "ParScanner.h"

#include "GenericToolbox.VariablesMonitor.h"
#include "GenericToolbox.CycleTimer.h"

#include "TDirectory.h"
#include "Math/Functor.h"
#include "Math/Minimizer.h"
#include "nlohmann/json.hpp"

#include "string"
#include "vector"
#include "memory"


class FitterEngine : public JsonBaseClass {

public:
  // Setters
  void setSaveDir(TDirectory *saveDir);
  void setEnablePreFitScan(bool enablePreFitScan);
  void setEnablePostFitScan(bool enablePostFitScan);
  void setEnablePca(bool enablePca_);

  // Getters
  double getChi2Buffer() const;
  double getChi2StatBuffer() const;
  double getChi2PullsBuffer() const;
  const Propagator& getPropagator() const;
  Propagator& getPropagator();
  ParScanner& getParScanner(){ return _parScanner_; }
  MinimizerInterface& getMinimizer(){ return _minimizer_; }
  TDirectory* getSaveDir(){ return _saveDir_; }

  double* getChi2BufferPtr(){ return &_chi2Buffer_; }
  double* getChi2StatBufferPtr(){ return &_chi2StatBuffer_; }
  double* getChi2PullsBufferPtr(){ return &_chi2PullsBuffer_; }
  double* getChi2RegBufferPtr(){ return &_chi2RegBuffer_; }

  // Core
  void fit();
  void updateChi2Cache();

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

  void fixGhostFitParameters();
  void rescaleParametersStepSize();
  void checkNumericalAccuracy();


private:
  // Parameters
  bool _enablePca_{false};
  bool _throwMcBeforeFit_{false};
  bool _enablePreFitScan_{false};
  bool _enablePostFitScan_{false};
  bool _scaleParStepWithChi2Response_{false};
  bool _debugPrintLoadedEvents_{false};
  int _debugPrintLoadedEventsNbPerSample_{10};
  double _throwGain_{1.};
  double _parStepGain_{0.1};

  // Internals
  TDirectory* _saveDir_{nullptr};
  Propagator _propagator_{};
  ParScanner _parScanner_{};
  MinimizerInterface _minimizer_{};

  // Buffers
  double _chi2Buffer_{0};
  double _chi2StatBuffer_{0};
  double _chi2PullsBuffer_{0};
  double _chi2RegBuffer_{0};

};


#endif //GUNDAM_FITTERENGINE_H
