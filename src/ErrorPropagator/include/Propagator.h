//
// Created by Nadrino on 11/06/2021.
//

#ifndef GUNDAM_PROPAGATOR_H
#define GUNDAM_PROPAGATOR_H


#include "DataSetLoader.h"
#include "PlotGenerator.h"
#include "EventTreeWriter.h"
#include "FitSampleSet.h"
#include "FitParameterSet.h"
#include "AnaSample.hh"

#include "GenericToolbox.CycleTimer.h"

#include "json.hpp"

#include <vector>
#include <future>


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
  const EventTreeWriter &getTreeWriter() const;

  // Core
  void propagateParametersOnSamples();
  void updateDialResponses();
  void reweightSampleEvents();
  void refillSampleHistograms();
  void applyResponseFunctions();

  // Switches
  void preventRfPropagation();
  void allowRfPropagation();

  // Monitor

  // Dev
  void fillDialsStack();

protected:
  void initializeThreads();

  void makeResponseFunctions();

  // multi-threaded
  void updateDialResponses(int iThread_);
  void reweightSampleEvents(int iThread_);
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
  EventTreeWriter _treeWriter_;
  std::vector<FitParameterSet> _parameterSetsList_;
  std::vector<DataSetLoader> _dataSetList_;
  std::shared_ptr<TMatrixD> _globalCovarianceMatrix_;

  // Response functions (WIP)
  std::map<FitSample*, std::shared_ptr<TH1D>> _nominalSamplesMcHistogram_;
  std::map<FitSample*, std::vector<std::shared_ptr<TH1D>>> _responseFunctionsSamplesMcHistogram_;

  // DEV
  std::vector<Dial*> _dialsStack_;


public:
  GenericToolbox::CycleTimer dialUpdate;
  GenericToolbox::CycleTimer weightProp;
  GenericToolbox::CycleTimer fillProp;
  GenericToolbox::CycleTimer applyRf;

  long long nbWeightProp = 0;
  long long cumulatedWeightPropTime = 0;

};


#endif //GUNDAM_PROPAGATOR_H
