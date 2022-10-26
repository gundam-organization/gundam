//
// Created by Nadrino on 11/06/2021.
//

#ifndef GUNDAM_PROPAGATOR_H
#define GUNDAM_PROPAGATOR_H


#include "DatasetLoader.h"
#include "PlotGenerator.h"
#include "EventTreeWriter.h"
#include "FitSampleSet.h"
#include "FitParameterSet.h"
#include "JsonBaseClass.h"

#include "GenericToolbox.CycleTimer.h"

#include <vector>
#include <map>
#include <future>

class Propagator : public JsonBaseClass {

public:
  // Setters
  void setSaveDir(TDirectory *saveDir);
  void setShowTimeStats(bool showTimeStats);
  void setThrowAsimovToyParameters(bool throwAsimovToyParameters);
  void setIThrow(int iThrow);
  void setLoadAsimovData(bool loadAsimovData);

  // Getters
  bool isThrowAsimovToyParameters() const;
  FitSampleSet &getFitSampleSet();
  std::vector<FitParameterSet> &getParameterSetsList();
  const std::vector<FitParameterSet> &getParameterSetsList() const;
  PlotGenerator &getPlotGenerator();
  const EventTreeWriter &getTreeWriter() const;

  int getIThrow() const;

  // Core
  void propagateParametersOnSamples();
  void updateDialResponses();
  void reweightMcEvents();
  void refillSampleHistograms();
  void applyResponseFunctions();

  // Dev
  void fillDialsStack();

  void generateSamplePlots(const std::string& savePath_, TDirectory* baseDir_ = nullptr);

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

  void initializeThreads();

  // multithreading
  void updateDialResponses(int iThread_);
  void reweightMcEvents(int iThread_);
  void applyResponseFunctions(int iThread_);

private:
  // Parameters
  bool _showTimeStats_{false};
  bool _loadAsimovData_{false};
  TDirectory* _saveDir_{nullptr};

  // Internals
  bool _throwAsimovFitParameters_{false};
  bool _throwAsimovToyParameters_{false};
  bool _enableStatThrowInToys_{true};
  bool _enableEventMcThrow_{true};
  int _iThrow_{-1};
  FitSampleSet _fitSampleSet_;
  PlotGenerator _plotGenerator_;
  EventTreeWriter _treeWriter_;
  std::vector<FitParameterSet> _parameterSetsList_;
  std::vector<DatasetLoader> _dataSetList_;
  std::shared_ptr<TMatrixD> _globalCovarianceMatrix_;

  // Monitoring
  bool _showEventBreakdown_{true};

  // Response functions (WIP)
  std::map<FitSample*, std::shared_ptr<TH1D>> _nominalSamplesMcHistogram_;
  std::map<FitSample*, std::vector<std::shared_ptr<TH1D>>> _responseFunctionsSamplesMcHistogram_;

  // DEV
  std::vector<Dial*> _dialsStack_;

#ifdef GUNDAM_USING_CACHE_MANAGER
  // Build the precalculated caches.  This is only relevant when using a GPU
  // and must be done after the datasets are loaded.  This returns true if
  // the cache is built.
  bool buildGPUCaches();

  // Prefill the caches using a GPU (if available).  If the GPU is not
  // available, then this is a NOP.  This copies the fit parameter values into
  // the GPU, triggers the appropriate kernel(s), and copies the results into
  // a CPU based cache.  This returns true if the cache is filled.
  bool fillGPUCaches();

  // A map of parameters to the indices that got used by the GPU.
  std::map<const FitParameter*, int> _gpuParameterIndex_;
#endif

public:
  GenericToolbox::CycleTimer dialUpdate;
  GenericToolbox::CycleTimer weightProp;
  GenericToolbox::CycleTimer fillProp;
  GenericToolbox::CycleTimer applyRf;

  long long nbWeightProp = 0;
  long long cumulatedWeightPropTime = 0;

};


#endif //GUNDAM_PROPAGATOR_H
