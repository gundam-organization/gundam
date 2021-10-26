//
// Created by Nadrino on 11/06/2021.
//

#ifndef XSLLHFITTER_PROPAGATOR_H
#define XSLLHFITTER_PROPAGATOR_H

#include "vector"
#include "future"

#include "json.hpp"

#include "GenericToolbox.CycleTimer.h"

#include "FitParameterSet.h"
#include "AnaSample.hh"
#include "PlotGenerator.h"
#include "FitSampleSet.h"
#include "DataSet.h"

class Propagator {

public:
  Propagator();
  virtual ~Propagator();

  // Initialize
  void reset();

  // Setters
  void setShowTimeStats(bool showTimeStats);
  void setSaveDir(TDirectory *saveDir);
  void setConfig(const json &config);

  // Init
  void initialize();

  // Getters
  bool isUseResponseFunctions() const;
  FitSampleSet &getFitSampleSet();
  std::vector<FitParameterSet> &getParameterSetsList();
  PlotGenerator &getPlotGenerator();
  const json &getConfig() const;

  // Core
  void propagateParametersOnSamples();
  void reweightSampleEvents();
  void refillSampleHistograms();
  void applyResponseFunctions();

  // Switches
  void preventRfPropagation();
  void allowRfPropagation();

  // Monitor

protected:
  void initializeThreads();
  void initializeCaches();

  void fillEventDialCaches();
  void makeResponseFunctions();

  // multi-threaded
  void reweightSampleEvents(int iThread_);
  void fillEventDialCaches(int iThread_);
  void applyResponseFunctions(int iThread_);

private:
  // Parameters
  bool _showTimeStats_{false};
  TDirectory* _saveDir_{nullptr};
  nlohmann::json _config_;

  // Internals
  bool _isInitialized_{false};
  bool _useResponseFunctions_{false};
  bool _isRfPropagationEnabled_{false};
  FitSampleSet _fitSampleSet_;
  PlotGenerator _plotGenerator_;
  std::vector<FitParameterSet> _parameterSetsList_;
  std::vector<DataSet> _dataSetList_;
  std::shared_ptr<TMatrixD> _globalCovarianceMatrix_;

  // Response functions (WIP)
  std::map<FitSample*, std::shared_ptr<TH1D>> _nominalSamplesMcHistogram_;
  std::map<FitSample*, std::vector<std::shared_ptr<TH1D>>> _responseFunctionsSamplesMcHistogram_;


public:
  GenericToolbox::CycleTimer weightProp;
  GenericToolbox::CycleTimer fillProp;
  GenericToolbox::CycleTimer applyRf;

  long long nbWeightProp = 0;
  long long cumulatedWeightPropTime = 0;

};


#endif //XSLLHFITTER_PROPAGATOR_H
